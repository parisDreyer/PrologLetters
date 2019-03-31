:- consult("utilities/relations").
:- consult("data").

answer(Question) :-
    not(log_if_present(Question)),
    log_to_data(Question).

log_to_data(Question) :-
    open('data.pl', append, Stream),
      write(Stream,  "data("),
        write(Stream, Question),
        write(Stream, ').'),
      nl(Stream),
      close(Stream),
    write("\nlogged data"),
    consult("data").

log_if_present([H|T]) :-
  is_data(H) ;
  log_if_present(T).


is_data(X) :-
  data(Y),
  member(X, Y),
  write(Y).

is_data(X) :-
  data(Y),
  subset_with_replacement(X, Y),
  write(Y).
