:- consult("utilities/list_operations").
:- consult("utilities/relations").
:- consult("data").

answer(Question, Explanation) :-
  try_explain(Question, Explanation)
  ;
  log_to_data(Question, Explanation).

without_last([X|Xs], [X|WithoutLast]) :-
  without_last(Xs, WithoutLast).

try_explain(Question, Explanation) :-
  already_logged(Question, Out),
  flatten(Out, Flat),
  without_last(Flat, FlatWithoutLast),
  write(Flat),
  write(FlatWithoutLast),
  atomize(Explanation, FlatWithoutLast).

log_to_data(Question, Explanation) :-
  log_to_data(Question),
  atomize(Question, Explanation).

log_to_data(Question) :-
    open('data.pl', append, Stream),
      write(Stream,  "data("),
        write(Stream, Question),
        write(Stream, ').'),
      nl(Stream),
      close(Stream),
    consult("data").

already_logged([H|T], [ExplanationItem|RestOfExplanation]) :-
  is_data(H, ExplanationItem)
  ;
  already_logged(T, RestOfExplanation).

% data(X, Y) :- data(Z), X = Z, Y = X.
data(X, Y) :- data(X), Y = X.

is_data(X, Explanation) :-
  data(Y, Explanation),
  member(X, Y).

is_data(X, Explanation) :-
  data(Y, Explanation),
  subset_with_replacement(X, Y).
