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
	// Load the other party's public key from a file
	remotePubKeyHex, err := ioutil.ReadFile("public")
	if err != nil {
		log.Fatal(err)
	}
	//fmt.Println("remotePubKeyHex:", string(remotePubKeyHex))

	// Convert the hex public key to bytes
	remotePubKeyBytes, err := hex.DecodeString(string(remotePubKeyHex))
	if err != nil {
		log.Fatal(err)
	}
	//fmt.Println("remotePubKeyBytes: ", remotePubKeyBytes)
	//fmt.Printf("%T\n", remotePubKeyBytes)

	remotePubKey := createECPublicKey(remotePubKeyBytes)
	fmt.Println("remotePubKey: ", remotePubKey)
	fmt.Printf("%T\n", remotePubKey)

	///////////////////////////////////////////////////////////////6curve

    // Generate your private key
	privKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("privKey: ", privKey)
    fmt.Printf("%T\n", privKey)


    sharedX, sharedY := elliptic.P256().ScalarMult(remotePubKey.X, remotePubKey.Y, privKey.D.Bytes())
    fmt.Println("sharedX, sharedY: ", sharedX, sharedY)
    fmt.Printf("%T%T\n", sharedX, sharedY)

    sharedSecret := new(big.Int).SetBytes(x.Bytes())
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
