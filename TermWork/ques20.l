%{
    #include<stdio.h>
    #include<string.h>
%}
%%
[a-zA-Z0-9 ]* {
    int n = yyleng;
    int org = 0;
    for(int i = 0;i < n;i++){
        org = org*10 + (yytext[i]-'0');
    }
    int temp = org*org;
    if(org%10 == temp%10){
        printf("%s is an Automorphic Number\n",yytext);
    }
    else{
        printf("%s is not an Automorphic Number\n",yytext);
    }
}
%%
int yywrap(){
    return 1;
}
int main(){
    printf("Enter the Input: \n");
    yylex();
}