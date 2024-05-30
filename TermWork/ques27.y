%{
#include <stdio.h>
#include <stdlib.h>
%}

%token A B

%%

S:
    A A A A A M B { printf("Valid string\n"); }
    ;

M:
    /* empty */
    | M A
    ;

%%

int main(void) {
    return yyparse();
}

void yyerror() {
    printf("Invalid\n");
    exit(0);
}
