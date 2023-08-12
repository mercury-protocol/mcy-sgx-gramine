package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"fmt"
	"log"
)

func main() {
	// Generate your private key and public key
	privKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatal(err)
	}

	// Marshal the public key to send to the other party
	pubKeyBytes := elliptic.Marshal(privKey.Curve, privKey.PublicKey.X, privKey.PublicKey.Y)

	// Simulate receiving the other party's public key
	// In reality, you would receive this from the other party
	otherPartyPubKey, _ := ecdsa.Unmarshal(elliptic.P256(), pubKeyBytes)

	// Perform ECDH key exchange
	sharedX, _ := privKey.Curve.ScalarMult(otherPartyPubKey.X, otherPartyPubKey.Y, privKey.D.Bytes())
	sharedSecret := sharedX.Bytes()

	fmt.Println("Shared Secret:", sharedSecret)
}
