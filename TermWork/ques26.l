%{
#include "y.tab.h"

%}

%%
a    { return A; }
b    { return B; }
[ \t\n]    { /* ignore whitespace and newlines */ }
.    { yyerror("Invalid character"); }

%%

int yywrap(void) {
    return 1;
}
