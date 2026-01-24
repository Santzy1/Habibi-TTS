dialect_id_map = {
    "UNK": "⓪",
    "MSA": "①",
    "SAU": "②",
    "UAE": "③",
    "ALG": "④",
    "IRQ": "⑤",
    "EGY": "⑥",
    "MAR": "⑦",
    "OMN": "⑧",
    "TUN": "⑨",
    "LEV": "⑩",
    "SDN": "⑪",
    "LBY": "⑫",
}  # use by infer_gradio and infer_cli


def text_list_formatter(text_list, dialect_id=None):
    final_text_list = []

    for text in text_list:
        if dialect_id is not None:
            text = f"{dialect_id}〈{text}〉"
        final_text_list.append(text)

    return final_text_list
