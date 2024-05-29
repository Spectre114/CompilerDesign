%{
    #include<stdio.h>
%}
%%
[^@]+"@"["gmail"|"hotmail"]+"."["com"|"org"]+ {
    printf("%s is a valid email\n",yytext);
}
[^\n]+ {
    printf("%s is an Invalid Email\n",yytext);
}
%%
int yywrap(){
    return 1;
}
int main(){
    printf("Enter the Input: \n");
    yylex();
}