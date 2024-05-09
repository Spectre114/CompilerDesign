%{
int c = 0;
%}
%%
[a-z] {c++;}
[A-Z] {c++;}
[ ] {printf("space\n");}
[\n] {printf("new line\n");}
. {printf("invalid\n");}
%%
yywrap() {return 1;}
int main()
{
	printf("Enter your Inputs: ");
	yylex();
	printf("Total chars: %d\n",c);
	return 0;
}


