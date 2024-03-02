%{
    int newLine = 0,space = 0,c = 0;
%}
%%
[a-z] {c++;}
[A-Z] {c++;}
[ ] {space++;}
[\n] {newLine++;}
%%
yywrap() {return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
    printf("No. of chars: %d\n",c);
    printf("No. of spaces: %d\n",space);
    printf("No. of new lines: %d\n",newLine);
    printf("No. of words: %d\n",space + newLine);w
}