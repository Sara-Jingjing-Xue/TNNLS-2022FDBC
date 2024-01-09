function F=turnfF(f,d,c)
F=zeros(d,c);
  for i=1:d
    F(i,f(i))=1;
  end
