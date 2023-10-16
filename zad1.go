package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"
)

func CreateFile() {
	fmt.Println("Writing to a file in Go lang")

	rand.Seed(time.Now().UnixNano())

	countPoint := 12
	countCats := 3
	minx := 0.0
	miny := 0.0
	maxx := 50.0
	maxy := 50.0
	route := []string{}

	file, err := os.Create("format.txt")
	if err != nil {
		log.Fatalf("Failed creating file: %s", err)
	}
	defer file.Close()

	for i := 0; i < countPoint; i++ {
		x1 := rand.Float64()*(maxx-minx) + minx
		y1 := rand.Float64()*(maxy-miny) + miny

		_, err := fmt.Fprintf(file, "%.2f$%.2f!", x1, y1)

		if err != nil {
			log.Fatalf("Failed writing to file: %s", err)
		}

		route = append(route, fmt.Sprintf("%.2f$%.2f!", x1, y1))
	}

	mop, err := file.WriteString("\n##\n")
	if err != nil {
		log.Fatalf("Failed writing to file: %s", err)
		fmt.Println(mop)
	}

	for k := 0; k < countCats; k++ {
		for i := 0; i < len(route); i++ {
			randomCount := len(route)
			randomIndex := rand.Intn(randomCount)
			pick := route[randomIndex]

			_, err := fmt.Fprintf(file, pick)

			if err != nil {
				log.Fatalf("Failed writing to file: %s", err)
			}
		}
		mop, err := file.WriteString("E")
		if err != nil {
			log.Fatalf("Failed writing to file: %s", err)
			fmt.Println(mop)
		}
	}

	fmt.Printf("\nFile Name: %s", file.Name())
}

func ReadFile() {
	fmt.Println("\n\nReading a file in Go lang")
	fileName := "format.txt"

	data, err := os.ReadFile(fileName)
	if err != nil {
		log.Fatalf("Failed reading data from file: %s", err)
	}

	fmt.Printf("\nFile Name: %s", fileName)
	fmt.Printf("\nSize: %d bytes", len(data))
	fmt.Printf("\nData: %s", data)
}

func main() {
	CreateFile()
	ReadFile()
}
