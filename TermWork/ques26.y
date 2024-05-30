%{
    #include<stdio.h>
    #include<stdlib.h>
%}
%token A B NL;
%%
St: S NL {
    printf("Valid Input\n");
    exit(0);
}
S: A S B | A B;
%%
int main(){
    yyparse();
    return 0;
}
int yyerror(){
    printf("Invalid\n");
    exit(0);
}