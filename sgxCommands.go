package main

import "fmt"
import "os"
import "syscall"


func executeCmd(args ...string) {
    cmd := os.exec.Command(args[0], args[1:]...)
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



func main() {
    startup()
    //executeCmd("python3", "-c", "import main; main.startup()")
}
