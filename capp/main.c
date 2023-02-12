/*
make
gramine-sgx ./main
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define QUOTE_PATH "/dev/attestation/quote"


int write(void)
{
    //char *filename = "test.txt";
    char *filename = "/dev/addestation/keys/testkey";

    // open the file for writing
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error opening the file %s\n", filename);
        return -1;
    }
    // write to the text file
    for (int i = 0; i < 10; i++)
        fprintf(fp, "This is the line #%d\n", i + 1);

    // close the file
    fclose(fp);

    return 0;
}


int read(void)
{
    FILE* ptr;
    char ch;
 
    // Opening file in reading mode
    //ptr = fopen("test.txt", "r");
    //ptr = fopen("dev/attestation/quote", "r");
    ptr = fopen("/dev/addestation/keys/testkey", "r");
 
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


int main(void)
{
    write();
    read();
}
