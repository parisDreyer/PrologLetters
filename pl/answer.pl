:- consult("utilities/list_operations").
:- consult("utilities/relations").
:- consult("data").

answer(Question, Explanation) :-
    not(log_if_present(Question, Explanation)),
    log_to_data(Question, Explanation).

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

log_if_present([H|T], [ExplanationItem|RestOfExplanation]) :-
  is_data(H, ExplanationItem) ;
  log_if_present(T, RestOfExplanation).

data(X, Y) :- data(Z), X = Z, Y = X.

is_data(X, Explanation) :-
  data(Y, Explanation),
  member(X, Y)
  .

is_data(X, Explanation) :-
  data(Y, Explanation),
  subset_with_replacement(X, Y)
  .
