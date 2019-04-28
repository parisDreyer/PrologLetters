% https://stackoverflow.com/questions/22659647/loading-a-prolog-file-with-arguments-from-windows-command-line

main :-
    current_prolog_flag(argv, AllArgs),
    print("Writing Args"),
    write(AllArgs),
    append(_, [-- | Args], AllArgs),
    write(Args),
    halt.


% Run with:
%
% $ swipl -s logic1.pl -t main --quiet -- folder1
% [folder1]
