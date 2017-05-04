package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"log"
	"math/rand"
	"net/http"
	"sort"
	"time"

	"golang.org/x/crypto/acme/autocert"

	"github.com/orijtech/otils"

	expirable "github.com/odeke-em/cache"
	"github.com/odeke-em/kmeans"
	"github.com/odeke-em/usgs"
)

var errUnimplemented = errors.New("unimplemented")

type feature struct {
	usgs.Feature
	memoizedSignature string
}

var _ kmeans.Vector = (*feature)(nil)

func (f *feature) Signature() interface{} {
	if f.memoizedSignature != "" {
		return f.memoizedSignature
	}
	f.memoizedSignature = fmt.Sprintf("%f-%f-%f-%f",
		f.Geometry.Coordinates.Latitude,
		f.Geometry.Coordinates.Longitude,
		f.Geometry.Coordinates.Depth,
		f.Properties.Magnitude,
	)
	return f.memoizedSignature
}

func (f *feature) Len() int { return 1 }
func (f *feature) Dimension(i int) (interface{}, error) {
	switch i {
	case 0:
		return f.Properties.Magnitude, nil
	default:
		return nil, errUnimplemented
	}
}

// Transformer for a feature --> Geometry.Coordinates

func earthquakeFeaturesToKmeans(features []*usgs.Feature) []kmeans.Vector {
	var vectors []kmeans.Vector
	for _, feat := range features {
		f := feature{*feat, ""}
		vectors = append(vectors, &f)
	}

	return vectors
}

type DumpElement struct {
	Color string `json:"color"`

	Centroid *usgs.Feature   `json:"centroid"`
	Points   []*usgs.Feature `json:"points"`
}

var tmpl = template.Must(template.New("index.html").ParseFiles("./static/index.html"))

const day = 24 * time.Hour

func durationLevel(durStr string) (period usgs.Period, err error) {
	fetchDur, err := time.ParseDuration(durStr)
	if err != nil {
		return period, err
	}

	if fetchDur <= time.Hour {
		return usgs.PastHour, nil
	}
	if fetchDur <= 1*day {
		return usgs.PastDay, nil
	}
	if fetchDur <= 7*day {
		return usgs.Past7Days, nil
	}

	return usgs.Past30Days, nil
}

type render struct {
	Period    string
	Elements  []*DumpElement
	LegendMap map[string]float32
}

var cacheableTimeDurations = map[usgs.Period]time.Duration{
	usgs.Past30Days: 1 * day, // Make a sliding window
	usgs.Past7Days:  1 * day,
	usgs.PastDay:    6 * time.Hour,
	usgs.PastHour:   1 * time.Hour,
}

func searchAndRender(rw http.ResponseWriter, req *http.Request) {
	query := req.URL.Query()
	usgsPeriod, err := durationLevel(query.Get("dur"))
	if err != nil {
		usgsPeriod = usgs.Past7Days
	}

	ureq := &usgs.Request{
		Period:    usgsPeriod,
		Magnitude: usgs.MAll,
	}
	elems, err := lookupEarthquakes(ureq)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}

	legendMap := make(map[string]float32)
	for _, dElem := range elems {
		centroid := dElem.Centroid
		legendMap[dElem.Color] = centroid.Properties.Magnitude
	}

	sort.Slice(elems, func(i, j int) bool {
		ei, ej := elems[i], elems[j]
		return ei.Centroid.Properties.Magnitude < ej.Centroid.Properties.Magnitude
	})

	rd := &render{
		Period:    fmt.Sprintf("Earthquakes in the past %s", usgsPeriod),
		Elements:  elems,
		LegendMap: legendMap,
	}

	if err := tmpl.Execute(rw, rd); err != nil {
		log.Printf("err rendering: %v", err)
	}
}

var client *usgs.Client

var commonSeed int64

func init() {
	commonSeed = time.Now().UnixNano()
	rand.Seed(commonSeed)

	var err error
	client, err = usgs.NewClient()
	if err != nil {
		log.Fatalf("initializing usgs client: %v", err)
	}
}

func minOf(args ...int) int {
	min := args[0]
	for _, arg := range args {
		if arg < min {
			min = arg
		}
	}
	return min
}

var expiryCache = expirable.New()

type dumpElementCacheExpirable struct {
	elems      []*DumpElement
	expiryTime time.Time
}

var _ expirable.Expirable = (*dumpElementCacheExpirable)(nil)

func (dec *dumpElementCacheExpirable) Expired(now time.Time) bool {
	return now.After(dec.expiryTime)
}

func (dec *dumpElementCacheExpirable) Value() interface{} {
	return dec.elems
}

type asyncClusters struct {
	err   error
	elems []*DumpElement
}

func lookupEarthquakes(usgsReq *usgs.Request) ([]*DumpElement, error) {
	asyncLookup := func(ureq *usgs.Request) chan *asyncClusters {
		cChan := make(chan *asyncClusters)
		go func() {
			defer close(cChan)
			elems, err := doLookupThenCache(ureq)
			cChan <- &asyncClusters{err: err, elems: elems}
		}()
		return cChan
	}

	ctx, cancel := context.WithTimeout(context.Background(), maxDeadline)
	defer cancel()

	var cluster *asyncClusters

	select {
	case cluster = <-asyncLookup(usgsReq):
	case <-ctx.Done():
		return nil, errTimedOut
	}

	return cluster.elems, cluster.err
}

var maxDeadline = 30 * time.Second
var errTimedOut = errors.New("😢 timed out!")

func doLookupThenCache(usgsReq *usgs.Request) ([]*DumpElement, error) {
	retr, ok := expiryCache.Get(usgsReq.Period)
	if ok {
		dumpSave := retr.(*dumpElementCacheExpirable)
		return dumpSave.elems, nil
	}

	// Otherwise a cache miss
	clusteredElems, err := _lookupEarthquakes(usgsReq)
	if err != nil {
		return nil, err
	}

	// Time to cache the results for the period of validity
	timeDuration := cacheableTimeDurations[usgsReq.Period]
	expiryPeriod := time.Now().Add(timeDuration)
	deSave := &dumpElementCacheExpirable{clusteredElems, expiryPeriod}
	expiryCache.Put(usgsReq.Period, deSave)
	log.Printf("Performed cache set: %#v for period: %v", usgsReq, timeDuration)

	return clusteredElems, nil
}

func _lookupEarthquakes(usgsReq *usgs.Request) ([]*DumpElement, error) {
	usgsResp, err := client.Request(usgsReq)
	if err != nil {
		return nil, err
	}

	// Goal is to group Feature items by magnitude?
	// Transform magnitues to vectors
	earthquakeVectors := earthquakeFeaturesToKmeans(usgsResp.Features)

	km := &kmeans.KMean{
		K: minOf(8, len(earthquakeVectors)/2),

		Vectors: earthquakeVectors,
		Seed:    commonSeed,
	}

	clusters, err := kmeans.KMeanify(km)
	if err != nil {
		return nil, err
	}

	elementsList := make([]*DumpElement, 0, len(clusters))
	randColor := randomColorFn()
	for key, points := range clusters {
		color := randColor()
		keyFeature := key.(*feature)

		var kFeatures []*usgs.Feature
		for _, pt := range points {
			kFeatures = append(kFeatures, featureToKmeansFeature(pt))
		}
		dElem := &DumpElement{
			Color:    color,
			Centroid: &(keyFeature.Feature),
			Points:   kFeatures,
		}
		elementsList = append(elementsList, dElem)
	}

	return elementsList, nil
}

func main() {
	var http1 bool
	flag.BoolVar(&http1, "http1", false, "run it as an HTTP1 server")
	flag.Parse()

	mux := http.NewServeMux()

	mux.HandleFunc("/visual", searchAndRender)
	mux.Handle("/", http.FileServer(http.Dir("./static")))

	if http1 {
		addr := ":8888"
		log.Printf("server on addr: %q", addr)
		if err := http.ListenAndServe(addr, mux); err != nil {
			log.Fatalf("serving err: %v", err)
		}
		return
	}

	go func() {
		nonHTTPSHandler := otils.RedirectAllTrafficTo("https://earthquakes.orijtech.com")
		if err := http.ListenAndServe(":80", nonHTTPSHandler); err != nil {
			log.Fatal(err)
		}
	}()

	domains := []string{
		"earthquakes.orijtech.com",
		"www.earthviz.orijtech.com",
		"earthviz.orijtech.com",
		"www.earthquakes.orijtech.com",
	}

	log.Fatal(http.Serve(autocert.NewListener(domains...), mux))
}

func featureToKmeansFeature(vect kmeans.Vector) *usgs.Feature {
	feat := vect.(*feature)
	return &feat.Feature
}

var mapboxColors = []string{
	"#000000",
	"#0000FF",
	"#00FF00",
	"#EE00EE",
	"#8a0a19",
	"#AA00FF",
	"#2e2e2d",
	"#130930",
	"#053307",
	"#380f05",
}

func randomColorFn() func() string {
	colors := mapboxColors[:]
	return func() string {
		i := rand.Intn(len(colors))
		if i >= len(colors) {
			return "#000000"
		}

		theColor := colors[i]
		smallerList := append([]string{}, colors[:i]...)
		smallerList = append(smallerList, colors[i+1:]...)
		colors = smallerList
		return theColor
	}
}
