%{
    #include"y.tab.h"
%}
%%
[a] {return A;}
[b] {return B;}
[\n] {return NL;}
%%
int yywrap(){
    return 1;
}