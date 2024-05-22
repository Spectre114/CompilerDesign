%{
    #include<stdio.h>
    #include<string.h>
    int minLen = 1e9;
    char word[100];
%}
%%
[^ \t\n]* {
    if(yyleng < minLen){
        minLen = yyleng;
        strcpy(word,yytext);
    }
}
%%
int yywrap(){
    return 1;
}
int main(){
    printf("Enter the Input: \n");
    yylex();
    printf("Shortest Word-> %s\n",word);
}