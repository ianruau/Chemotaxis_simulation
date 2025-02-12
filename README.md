# Simulations for Chemotaxis equations


# Stylish

Install black and isort and format the codes as follows
```bash
pip install black isort
sudo apt install autopep8
black simulation.py
isort simulation.py
autopep8 --in-place --aggressive simulation.py
```

Configure for nvim in case of init.vim:
```vim
autocmd BufWritePre *.py execute ':!black % && isort % && autopep8 --in-place --aggressive %'
```
or in case of init.lua
```lua
vim.api.nvim_create_autocmd("BufWritePre", {
    pattern = "*.py",
    command = "!black % && isort % && autopep8 --in-place --aggressive %",
})
```
