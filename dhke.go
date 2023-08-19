package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/ecdsa"
	"crypto/x509"
	"encoding/pem"
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
	curve := elliptic.P256()
	remotePubKeyX, remotePubKeyY := elliptic.Unmarshal(curve, remotePubKeyBytes)

	remotePubKey := ecdsa.PublicKey{
	    Curve: curve,
	    X:     remotePubKeyX,
	    Y:     remotePubKeyY,
	}

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

func parseECPublicKey(pubKeyBytes []byte) (*ecdsa.PublicKey, error) {
	// Create a block to hold the PEM-encoded public key
	block := &pem.Block{
		Type:  "EC PUBLIC KEY",
		Bytes: pubKeyBytes,
	}

	// Decode the PEM block into a DER-encoded public key
	derBytes, _ := pem.Decode(block.Bytes)
	if derBytes == nil {
		return nil, fmt.Errorf("failed to decode PEM block")
	}

	// Parse the DER-encoded public key into an ECDSA public key struct
	pubKey, err := x509.ParsePKIXPublicKey(derBytes.Bytes)
	if err != nil {
		return nil, err
	}

	// Convert to *ecdsa.PublicKey type
	ecdsaPubKey, ok := pubKey.(*ecdsa.PublicKey)
	if !ok {
		return nil, fmt.Errorf("unexpected public key type")
	}

	return ecdsaPubKey, nil
}

