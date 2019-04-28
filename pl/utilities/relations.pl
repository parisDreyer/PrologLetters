member(Item,[Item|_]). /* is the target item the head of the list */
member(Item,[_|Tail]) :- member(Item,Tail). /* signals to check rest of list with current selected item as new head for head being Item */
member([X1, Tail1], [X2, Tail2]) :-
  X1 == X2,
  member(Tail1, Tail2).
% member([], []).
% member(List, Set) :- subset_with_replacement(List, Set).

subset_with_replacement([], _).
subset_with_replacement([Head|Tail], Set) :-
  member(Head, Set),
  subset_with_replacement(Tail, Set).

letters_set([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]).
letters(List) :-
  letters_set(Letters),
  subset_with_replacement(List, Letters).

letter(X) :-
  letters_set(LettersSet),
  member(X, LettersSet).
