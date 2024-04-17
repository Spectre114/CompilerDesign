%{
    #include<stdio.h>
    int newLine = 0;
    int space = 0;
    int c = 0;
    int tabs = 0;
    int words = 0;
%}
%%
[ ] {space++;}
[\n] {newLine++;}
[\t] {tabs++;}
[^\n\t ]+ {words++;c += yyleng;} 
%%
yywrap() {return 1;}
int main(){
    printf("Enter input: ");
    yylex();
    printf("No. of characters: %d\n",c);
    printf("No. of spaces: %d\n",space);
    printf("No. of new lines: %d\n",newLine);
    printf("No. of words: %d\n",words);
    printf("No. of tabs: %d\n",tabs);
}