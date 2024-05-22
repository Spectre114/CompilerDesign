%{
    #include<stdio.h>
    #include<string.h>
%}
%%
[0-9]* {
    int n = yyleng;
    int flag = 1;
    for(int i = 0;i <= n/2;i++){
        if(yytext[i] != yytext[n-i-1]){
            flag = 0;
            break;
        }
    }
    if(flag){
        printf("%s is a Pallindrome\n",yytext);
    }
    else{
        printf("%s is not a Pallindrome\n",yytext);
    }
}
. {
    printf("Please enter number\n");
}
%%
int yywrap(){
    return 1;
}
int main(){
    printf("Enter the Input: \n");
    yylex();
}