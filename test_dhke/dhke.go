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

func loadRemotePublicKey() *ecdsa.PublicKey {
	remotePubKeyHex, err := ioutil.ReadFile("public_python")
	if err != nil {
		log.Fatal(err)
	}
	remotePubKeyBytes, err := hex.DecodeString(string(remotePubKeyHex))
	if err != nil {
		log.Fatal(err)
	}

	x := new(big.Int).SetBytes(remotePubKeyBytes[:32]) // X coordinate is the first 32 bytes
	y := new(big.Int).SetBytes(remotePubKeyBytes[32:]) // Y coordinate is the next 32 bytes

	return &ecdsa.PublicKey{
		Curve: elliptic.P256(),
		X:     x,
		Y:     y,
	}
}

func savePublicKey(privKey ecdsa.PrivateKey) {
	pubKeyBytes := append(privKey.PublicKey.X.Bytes(), privKey.PublicKey.Y.Bytes()...)
	pubKeyHex := hex.EncodeToString(pubKeyBytes)

	err := ioutil.WriteFile("public_go", []byte(pubKeyHex), 0644)
	if err != nil {
		log.Fatal(err)
	}
}

func generatePrivateKey() *ecdsa.PrivateKey {
	privKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatal(err)
	}

	return privKey
}

func generateSharedSecret(remotePubKey ecdsa.PublicKey, privKey ecdsa.PrivateKey) *big.Int {
    sharedX, _ := elliptic.P256().ScalarMult(remotePubKey.X, remotePubKey.Y, privKey.D.Bytes())
    sharedSecret := new(big.Int).SetBytes(sharedX.Bytes())

	return sharedSecret
}

func saveSharedSecret(sharedSecret big.Int) {
    sharedSecretHex := hex.EncodeToString(sharedSecret.Bytes())
	err := ioutil.WriteFile("shared_secret_go", []byte(sharedSecretHex), 0644)
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	remotePubKey := loadRemotePublicKey()
	fmt.Println("remotePubKey: ", remotePubKey)
	fmt.Printf("%T\n", remotePubKey)

	privKey := generatePrivateKey()
	fmt.Println("privKey: ", privKey)
    fmt.Printf("%T\n", privKey)

    savePublicKey(*privKey)

    sharedSecret := generateSharedSecret(*remotePubKey, *privKey)
    fmt.Println("sharedSecret:", sharedSecret)
	fmt.Printf("%T\n", sharedSecret)

	saveSharedSecret(*sharedSecret)  // for testing purposes only
}
