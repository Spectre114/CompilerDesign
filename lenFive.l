%{
    #include<stdio.h>
    int words = 0;
%}
%%
[^\t\n ]+ {
    int c = yyleng;
    if(c == 5){
        words++;
    }
}
%%
int yywrap(){
    return 1;
}
int main(){
    printf("Enter the input: \n");
    yylex();
    printf("%d\n",words);
}