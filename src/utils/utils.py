import textwrap

def get_step_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < total_steps * 0.7:
            return 1.0
        else:
            progress = (current_step - total_steps * 0.7) / (total_steps * 0.3)
            return max(0.1, 1.0 - 0.9 * progress)
    return lr_lambda

def freeze_layers(model, layers_to_freeze):
    for idx in layers_to_freeze:
        for param in model.decoder_blocks[idx].parameters():
            param.requires_grad = False
    
def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True

def log_progress(text):
    fixed_width = 82
    formatted_text = f"║ {text:<{fixed_width}} ║"
    print(formatted_text)

def render_chat_box(test_cases, models, generate_response, total_width=70, max_new_tokens=100, beam_size=5):
    INNER_WIDTH = total_width - 2
    PADDING = 1
    CONTENT_WIDTH = INNER_WIDTH - PADDING * 2
    SIDE_WIDTH = (CONTENT_WIDTH - 10) // 2

    def print_line(content=""):
        print("║" + " " * PADDING + f"{content:<{CONTENT_WIDTH}}" + " " * PADDING + "║")

    def print_user(text):
        # Tên user căn lề phải
        print_line(f"{'User:':>{CONTENT_WIDTH}}")
        wrapped = textwrap.wrap(text, SIDE_WIDTH)
        for line in wrapped:
            print_line(f"{line:>{CONTENT_WIDTH}}")

    def print_bot(model_name, text):
        print_line(f"{model_name}:")
        wrapped = textwrap.wrap(text, SIDE_WIDTH)
        for line in wrapped:
            print_line(line)

    print("╔" + "═" * INNER_WIDTH + "╗")

    for user_input in test_cases:
        print_user(user_input)
        for model_name, model_info in models.items():
            model = model_info["model"]
            response = generate_response(
                model,
                user_input,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
            )
            print_bot(model_name, response)

        print_line()
        print_line()

    print("╚" + "═" * INNER_WIDTH + "╝")
