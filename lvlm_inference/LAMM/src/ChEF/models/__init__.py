import torch

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip

def get_model(cfg, device):
    model_name = cfg['model_name']
    if model_name == 'InstructBLIP':
        from .test_instructblip import TestInstructBLIP
        return TestInstructBLIP(device=device)
    elif model_name == 'InstructBLIP-Trace':
        from .test_instructblip_trace import TestInstructBLIPTrace
        return TestInstructBLIPTrace(device=device)
    elif model_name == 'LLaMA-Adapter-v2':
        from .test_llamaadapterv2 import TestLLamaAdapterV2
        return TestLLamaAdapterV2(device=device, **cfg)
    elif model_name == 'MiniGPT-4':
        from .test_minigpt4 import TestMiniGPT4
        return TestMiniGPT4(device=device, **cfg)
    elif model_name == 'MiniGPT-4-v2-Trace':
        from .test_minigpt4v2_trace import TestMiniGPT4V2Trace
        return TestMiniGPT4V2Trace(device=device, **cfg)
    elif model_name == 'MiniGPT-4-v2':
        from .test_minigpt4v2 import TestMiniGPT4V2
        return TestMiniGPT4V2(device=device, **cfg)
    elif model_name == 'mPLUG-Owl':
        from .test_mplugowl import TestMplugOwl
        return TestMplugOwl(device=device,**cfg)
    elif model_name == 'mPLUG-Owl2':
        from .test_mplugowl2 import TestMplugOwl2
        return TestMplugOwl2(device=device,**cfg)
    elif model_name == 'Otter':
        from .test_otter import TestOtter
        return TestOtter(device=device, **cfg)
    elif model_name == 'Kosmos2':
        from .test_kosmos import TestKOSMOS2
        return TestKOSMOS2(device=device,**cfg)
    elif model_name == 'LAMM':
        from .test_lamm import TestLAMM
        return TestLAMM(device=device, **cfg)
    elif model_name == 'LAMM_SFT' :
        from .test_lamm15 import TestLAMM15
        return TestLAMM15(device=device, **cfg)
    elif model_name == 'Octavius' or model_name == 'Octavius_3d' or model_name == 'Octavius_2d':
        from .test_octavius import TestOctavius
        return TestOctavius(**cfg)
    elif model_name == 'Shikra':
        from .test_shikra import TestShikra
        return TestShikra(device=device,**cfg)
    elif model_name == 'LLaVA1.5-Trace' or model_name == 'LLaVA1.5-13b-Trace' or model_name == 'LLaVA1.6-13b-Trace':
        from .test_llava15_trace import TestLLaVA15Trace
        return TestLLaVA15Trace(device=device, **cfg)
    elif model_name == 'LLaVA1.5' or model_name == 'LLaVA1.5-13b' or model_name == 'LLaVA1.6-13b':
        from .test_llava15 import TestLLaVA15
        return TestLLaVA15(device=device, **cfg)
    elif model_name == 'LLaVARLHF':
        from .test_llavarlhf import TestLLaVARLHF
        return TestLLaVARLHF(device=device, **cfg)
    elif model_name == 'LLaVARLHF-Trace':
        from .test_llavarlhf_trace import TestLLaVARLHFTrace
        return TestLLaVARLHFTrace(device=device, **cfg)
    elif model_name == 'InternLMXComposer-Trace':
        from .test_internlmxcomposer_trace import TestInternlmXcomposerTrace
        return TestInternlmXcomposerTrace(device=device, **cfg)
    elif model_name == 'InternLMXComposer':
        from .test_internlmxcomposer import TestInternlmXcomposer
        return TestInternlmXcomposer(device=device, **cfg)
    elif model_name == 'QwenVL':
        from .test_qwenvl import TestQwenVL
        return TestQwenVL(device=device, **cfg)
    elif model_name == 'Test':
        from .test_base import TestBase
        return TestBase(**cfg)
    elif model_name == 'RLHFV':
        from .test_rlhfv import TestRLHFV
        return TestRLHFV(device=device, **cfg)
    elif model_name == 'GPT':
        from .test_gpt import TestGPT
        return TestGPT(**cfg)
    elif model_name == 'Gemini':
        from .test_gemini import TestGemini
        return TestGemini(**cfg)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
