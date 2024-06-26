%{
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

void yyerror(const char *s);
int yylex(void);

%}

%token NUMBER
%left '+' '-'
%left '*' '/'
%right UMINUS

%%

St:  |St S
    ;

S:
    '\n'
    | expr '\n'  { printf("Result: %d\n", $1); }
    ;

expr:
    NUMBER          { $$ = $1; }
    | expr '+' expr  { $$ = $1 + $3; }
    | expr '-' expr  { $$ = $1 - $3; }
    | expr '*' expr  { $$ = $1 * $3; }
    | expr '/' expr  { $$ = $1 / $3; }
    | '-' expr %prec UMINUS { $$ = -$2; }
    | '(' expr ')'  { $$ = $2; }
    ;

%%

int main(void) {
    return yyparse();
}


