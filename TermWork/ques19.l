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
    int check = 0;
    while(temp){
        int rem = temp%10;
        check += rem;
        temp /= 10;
    }
    if(check == org){
        printf("%s is a Neon Number\n",yytext);
    }
    else{
        printf("%s is not a Neon Number\n",yytext);
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