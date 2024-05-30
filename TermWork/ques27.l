%{
#include "y.tab.h"
%}

%%

a    { return A; }
b    { return B; }
[ \t\n]    {}
.    { yyerror("Invalid character"); return 0; }

%%

int yywrap(void) {
    return 1;
}
