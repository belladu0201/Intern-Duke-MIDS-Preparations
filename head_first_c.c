/* intepret the C code */
/* We need to compile the code */
/* main() is necessary for any C code , it is important to start the code*/
# include <stdio.h> /* this makes it able to run in terminal */
int main() /* it returns an int in this program */
{
    int decks; /* store decks as an integer */
    puts("Enter a number of decks");  /* display the following*/
    scanf("%i", &decks); /* store everything of the user type into decks */
    if (decks < 1) {
    puts("That is not a valid number of decks");
    return 1; }
    /* return %i interger type following, the first parameter is decks*52 */
    /* first parameter will be inserted */
    printf("There are %i cards\n", (decks * 52)); /* display fomatted */
    return 0;

}
/* compliataion code: gcc head_first_c.c -o head_first_c*/
/* ./head_first_c */
/* one line compliation: gcc test.c -o test && ./test */
/* Done Page 10 */
/* we can't have two main functions in C */