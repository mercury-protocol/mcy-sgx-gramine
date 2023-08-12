package main

import (
	"crypto/rand"
	"crypto/diffiehellman"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"log"
	"math/big"
)

func main() {
	// Load the other party's public key from a file
	otherPartyPublicKeyHex, err := ioutil.ReadFile("other_party_public_key.txt")
	if err != nil {
		log.Fatal(err)
	}

	// Convert the hex public key to a big integer
	otherPartyPublicKey, success := new(big.Int).SetString(string(otherPartyPublicKeyHex), 16)
	if !success {
		log.Fatal("Invalid hexadecimal public key")
	}

	// Generate our private key and public key
	privKey, err := diffiehellman.GenerateKey(rand.Reader, 128)
	if err != nil {
		log.Fatal(err)
	}

	// Compute the shared secret
	sharedSecret := new(big.Int).Exp(otherPartyPublicKey, privKey, diffiehellman.P)
	sharedSecretBytes := sharedSecret.Bytes()

	// Hash the shared secret using SHA-256 to create a symmetric key
	hash := sha256.New()
	hash.Write(sharedSecretBytes)
	symmetricKey := hash.Sum(nil)

	fmt.Println("Symmetric Key:", hex.EncodeToString(symmetricKey))
}
