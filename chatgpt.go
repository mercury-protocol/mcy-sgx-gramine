package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"fmt"
	"log"
	"math/big"
)

func main() {
	// Generate an ECDSA private key and corresponding public key
	privKeyA, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatal(err)
	}

	privKeyB, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the shared point on the curve using party A's private key and party B's public key
	sharedX, sharedY := elliptic.P256().ScalarMult(privKeyB.PublicKey.X, privKeyB.PublicKey.Y, privKeyA.D.Bytes())

	// Create a shared public key using the shared point
	sharedPubKey := ecdsa.PublicKey{
		Curve: elliptic.P256(),
		X:     sharedX,
		Y:     sharedY,
	}

	// Calculate the shared secret (x-coordinate of the shared point)
	sharedSecret := sharedX.Bytes()

	fmt.Println("Shared Secret:", sharedSecret)
}
