%{
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    int newLine = 0;
    int space = 0;
    int c = 0;
    int words = 0;
%}
%%
[ ] {space++;}
[\n] {newLine++;}
[^\n\t ]+ {words++;c += yyleng;}
%%
int yywrap() {
    return 1;
}
int main(){
    extern FILE *yyin,*yyout;
    yyin=fopen("input.txt","r");
    yyout=fopen("output.txt","w");
    yylex();
    fprintf(yyout,"No. of chars: %d\n",c);
    fprintf(yyout,"No. of spaces: %d\n",space);
    fprintf(yyout,"No. of new lines: %d\n",newLine);
    fprintf(yyout,"No. of words: %d\n",words);
    fclose(yyin);
    fclose(yyout);
    return 0;
}

