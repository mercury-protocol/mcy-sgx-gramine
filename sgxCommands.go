package main

import "fmt"
import "os"
import "os/exec"
import "syscall"
import "encoding/json"
import "io/ioutil"


type AttestationReport struct {
    Signature string          `json:"X-IASReport-Signature"`
    SigningCertificate string `json:"X-IASReport-Signing-Certificate"`
    Body string               `json:"Body"`
}


func executeCmd(args ...string) {
    cmd := exec.Command(args[0], args[1:]...)
    fmt.Println(cmd.Args)
    out, err := cmd.CombinedOutput()
    if err != nil {
        fmt.Println(err);
    }
    fmt.Println(string(out))
}


func startup() {

    mydir, err := os.Getwd()
    if err != nil {
        fmt.Println(err)
    }
    mydir += "/app"

    syscall.Chdir(mydir)

    executeCmd("make", "distclean")
    executeCmd("make")
    executeCmd("gramine-sgx", "./sgxapp", "startup.py")
}


func remoteAttestation() AttestationReport {
    executeCmd("python3", "remote_attestation.py")

    jsonFile, err := os.Open("attestation_report.json")
    if err != nil {
        fmt.Println(err)
    }
    defer jsonFile.Close()

    byteValue, _ := ioutil.ReadAll(jsonFile)
    var attestationReport AttestationReport
    json.Unmarshal(byteValue, &attestationReport)

    return attestationReport
}


func trainModel() {
    executeCmd("gramine-sgx", "./sgxapp", "train_model.py")
}


func main() {
    startup()
    attestationReport := remoteAttestation()
    fmt.Println(attestationReport)
    trainModel()
}
