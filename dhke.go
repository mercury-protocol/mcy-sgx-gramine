package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"encoding/hex"
	"math/big"
	"fmt"
	"io/ioutil"
	"log"
)

func main() {
	// Load remote public key from file
	remotePubKeyHex, err := ioutil.ReadFile("public")
	if err != nil {
		log.Fatal(err)
	}

	// Convert the hex public key to bytes
	remotePubKeyBytes, err := hex.DecodeString(string(remotePubKeyHex))
	if err != nil {
		log.Fatal(err)
	}

	remotePubKey := createECPublicKey(remotePubKeyBytes)
	fmt.Println("remotePubKey: ", remotePubKey)
	fmt.Printf("%T\n", remotePubKey)

	///////////////////////////////////////////////////////////////6curve

    // Generate private key
	privKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("privKey: ", privKey)
    fmt.Printf("%T\n", privKey)

    // Save the public key in the same format as the Python script
	pubKeyBytes := append(privKey.PublicKey.X.Bytes(), privKey.PublicKey.Y.Bytes()...)
	pubKeyHex := hex.EncodeToString(pubKeyBytes)
	err = ioutil.WriteFile("public_go", []byte(pubKeyHex), 0644)
	if err != nil {
		log.Fatal(err)
	}

    // Generate shared secret
    sharedX, sharedY := elliptic.P256().ScalarMult(remotePubKey.X, remotePubKey.Y, privKey.D.Bytes())
    fmt.Println("sharedX, sharedY: ", sharedX, sharedY)
    fmt.Printf("%T%T\n", sharedX, sharedY)

    sharedSecret := new(big.Int).SetBytes(sharedX.Bytes())
	fmt.Println("sharedSecret:", sharedSecret)
}


func createECPublicKey(pubKeyBytes []byte) *ecdsa.PublicKey {
	curve := elliptic.P256()
	x := new(big.Int).SetBytes(pubKeyBytes[:32]) // X coordinate is the first 32 bytes
	y := new(big.Int).SetBytes(pubKeyBytes[32:]) // Y coordinate is the next 32 bytes

	return &ecdsa.PublicKey{
		Curve: curve,
		X:     x,
		Y:     y,
	}
}
