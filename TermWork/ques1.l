%{
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
int yywrap() {return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
    printf("No. of chars: %d\n",c);
    printf("No. of spaces: %d\n",space);
    printf("No. of new lines: %d\n",newLine);
    printf("No. of words: %d\n",words);
    printf("No. of tabs: %d\n",tabs);
}