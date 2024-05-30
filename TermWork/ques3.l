%{
    #include<stdio.h>
    int x,y,res;
%}
%%
[0-9 ]+"+"[ 0-9]+ {
    int n = yyleng;
    int x = 0;
    int y = 0;
    int i = 0;
    for(;yytext[i] != '+';i++){
        if(yytext[i] >= '0' && yytext[i] <= '9'){
            x = x*10 + yytext[i]-'0';
        }
    }
    for(;i<n;i++){
        if(yytext[i] >= '0' && yytext[i] <= '9'){
            y = y*10 + yytext[i]-'0';
        }
    }
    printf("%d",x+y);
}
[0-9]+"-"[0-9]+ {
    int n = yyleng;
    int x = 0;
    int y = 0;
    int i = 0;
    for(;yytext[i] != '-';i++){
        if(yytext[i] >= '0' && yytext[i] <= '9'){
            x = x*10 + yytext[i]-'0';
        }
    }
    for(;i<n;i++){
        if(yytext[i] >= '0' && yytext[i] <= '9'){
            y = y*10 + yytext[i]-'0';
        }
    }
    printf("%d",x-y);
}
[0-9]+"*"[0-9]+ {
    int n = yyleng;
    int x = 0;
    int y = 0;
    int i = 0;
    for(;yytext[i] != '*';i++){
        if(yytext[i] >= '0' && yytext[i] <= '9'){
            x = x*10 + yytext[i]-'0';
        }
    }
    for(;i<n;i++){
        if(yytext[i] >= '0' && yytext[i] <= '9'){
            y = y*10 + yytext[i]-'0';
        }
    }
    printf("%d",x*y);
}
[0-9]+"/"[1-9]+ {
    int n = yyleng;
    int x = 0;
    int y = 0;
    int i = 0;
    for(;yytext[i] != '/';i++){
        if(yytext[i] >= '0' && yytext[i] <= '9'){
            x = x*10 + yytext[i]-'0';
        }
    }
    for(;i<n;i++){
        if(yytext[i] >= '0' && yytext[i] <= '9'){
            y = y*10 + yytext[i]-'0';
        }
    }
    printf("%d",x/y);
}
%%
int yywrap(){
    return 1;
}
int main(){
    printf("Enter the expression to evaluate:\n");
    yylex();
    return 0;
}