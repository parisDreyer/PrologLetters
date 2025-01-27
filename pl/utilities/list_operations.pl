% atomize uses SWI-Prolog's inbuilt function atom_string(_, _) to convert
% a list of strings into a list of atoms
atomize([], []).
atomize([Word|Z], [Item|Tail]) :-
  atom_string(Word, Item),
  atomize(Z, Tail).
