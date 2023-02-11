/*
make
gramine-sgx ./main
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(void) {
    FILE* ptr;
    char ch;
 
    // Opening file in reading mode
    ptr = fopen("/dev/attestation/quote", "r");
 
    if (NULL == ptr) {
        printf("file can't be opened \n");
    }
 
    printf("content of this file are \n");
 
    // Printing what is written in file
    // character by character using loop.
    do {
        ch = fgetc(ptr);
        printf("%c", ch);
 
        // Checking if character is not EOF.
        // If it is EOF stop reading.
    } while (ch != EOF);

    printf("\n");
 
    // Closing the file
    fclose(ptr);
    return 0;
}
