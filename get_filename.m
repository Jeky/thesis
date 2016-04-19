function name = get_filename(f)
    index = strfind(f, '.');
    name = f(1:index(1)-1);
end