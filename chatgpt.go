package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"fmt"
	"log"
	"math/big"
)

func main() {
	// Bytes representing the ECDSA public key
	pubKeyBytes := []byte{
		0x04, // Uncompressed point format indicator
		0x5D, 0x5F, 0x38, 0xA3, 0x9C, 0x50, 0xD6, 0x97,
		// ... (more bytes representing X and Y coordinates)
	}

	// Create an ECDSA public key struct directly from the bytes
	pubKey := createECPublicKey(pubKeyBytes)
	if pubKey == nil {
		log.Fatal("Failed to create ECDSA public key")
	}

	fmt.Println("ECDSA Public Key:", pubKey)
}

func createECPublicKey(pubKeyBytes []byte) *ecdsa.PublicKey {
	// Construct an ECDSA public key struct directly
	curve := elliptic.P256() // Use the appropriate curve
	x := new(big.Int).SetBytes(pubKeyBytes[1 : curve.Params().BitSize/8+1])
	y := new(big.Int).SetBytes(pubKeyBytes[curve.Params().BitSize/8+1:])
	return &ecdsa.PublicKey{
		Curve: curve,
		X:     x,
		Y:     y,
	}
}
