def traduz_sentenca(model, tokenizer, texto_ingles):
    # Tokenize o texto único
    tokens = tokenizer(texto_ingles, return_tensors="pt", padding=True)

    # Gera a tradução com o modelo
    translated = model.generate(**tokens)

    # Decodifica a tradução gerada
    texto_portugues = tokenizer.decode(translated[0], skip_special_tokens=True)

    return {
        'texto_ingles': texto_ingles,
        'texto_portugues': texto_portugues
    }