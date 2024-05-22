%{
    #include<stdio.h>
    int v = 0;
    int c = 0;
%}
%%
[aeiouAEIOU] {
    v++;
    printf("Vowel-> %s\n",yytext);
}
[^ aeiouAEIOU] {
    c++;
    printf("Consonant %s\n",yytext);
}
%%
int yywrap(){
    return 1;
}
int main(){
    printf("Enter the input: \n");
    yylex();
    printf("Total Vowels -> %d\n",v);
    printf("Total Consonants -> %d\n",c);
}