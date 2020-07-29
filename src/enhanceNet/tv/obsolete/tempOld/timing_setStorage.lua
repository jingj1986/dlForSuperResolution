purpose = [[
Is it fast to set the storage value in lua?
]]

s = torch.LongStorage({1,2,3})
f = function(s)
    s[1] = 2
    s[2] = 4
    s[3] = 5

    s[2] = 2
    s[1] = 4
    s[3] = 5

    s[3] = 2
    s[2] = 4
    s[1] = 5

    s[1] = 5
end

nloop = 1000

time = torch.tic();
for i = 1, 1000 do
    f(s)
end
time = torch.toc(time)
print('time = ' .. time/nloop)
