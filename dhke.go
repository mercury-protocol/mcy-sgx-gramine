package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"log"
)

func main() {
	// Load the other party's public key from a file
	remotePubKeyHex, err := ioutil.ReadFile("remote_public_key")
	if err != nil {
		log.Fatal(err)
	}

	// Convert the hex public key to bytes
	remotePubKeyBytes, err := hex.DecodeString(string(remotePubKeyHex))
	if err != nil {
		log.Fatal(err)
	}

	// Unmarshal the remote public key
	remotePubKey, _ := ecdsa.Unmarshal(elliptic.P256(), remotePubKeyBytes)

	// Generate your private key
	privKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatal(err)
	}

	// Marshal your public key into bytes
	localPubKeyBytes := elliptic.Marshal(privKey.Curve, privKey.PublicKey.X, privKey.PublicKey.Y)

	// Perform ECDH key exchange
	sharedX, _ := privKey.Curve.ScalarMult(remotePubKey.X, remotePubKey.Y, privKey.D.Bytes())
	sharedSecret := sharedX.Bytes()

	// Print local public key in hex format
	fmt.Println("Local Public Key (Hex):", hex.EncodeToString(localPubKeyBytes))

	// Print the shared secret in hex format
	fmt.Println("Shared Secret (Hex):", hex.EncodeToString(sharedSecret))
}
