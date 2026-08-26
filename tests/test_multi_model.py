import pytest
from unittest.mock import Mock, MagicMock
from self_correct.core import AntiHallucinator

class MockClient:
    def __init__(self):
        self.calls = []
    
    class Completions:
        def __init__(self, parent):
            self.parent = parent
        
        def create(self, model, messages, temperature=0.2, max_tokens=None):
            self.parent.calls.append({
                'model': model,
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens
            })
            mock_resp = Mock()
            mock_resp.choices = [Mock(message=Mock(content='Test response'))]
            mock_resp.usage = Mock(prompt_tokens=10, completion_tokens=20)
            return mock_resp
    
    @property
    def chat(self):
        mock_chat = Mock()
        mock_chat.completions = self.Completions(self)
        return mock_chat

def test_multi_model_routing():
    '''Test that per-phase models are used correctly.'''
    client = MockClient()
    hallu = AntiHallucinator(
        client=client,
        strictness=1.0,
        model_draft='model-draft',
        model_extract='model-extract',
        model_verify='model-verify',
        model_correct='model-correct',
    )
    
    # Mock the extraction to return a simple claim
    original_call = client.chat.completions.create
    
    def mock_create(model, messages, **kwargs):
        # Check which phase we're in based on system prompt
        system_prompt = messages[0]['content'] if messages else ''
        client.calls.append({'model': model, 'system': system_prompt[:50]})
        
        mock_resp = Mock()
        mock_resp.choices = [Mock(message=Mock(content='Test claim 1'))]
        mock_resp.usage = Mock(prompt_tokens=10, completion_tokens=20)
        return mock_resp
    
    client.chat.completions.create = mock_create
    
    # We need to also mock the verification and correction phases
    call_count = [0]
    
    def mock_create_full(model, messages, **kwargs):
        call_count[0] += 1
        system_prompt = messages[0]['content'] if messages else ''
        
        mock_resp = Mock()
        if 'helpful assistant' in system_prompt:
            # Drafting phase
            mock_resp.choices = [Mock(message=Mock(content='Draft response'))]
        elif 'Extract' in system_prompt:
            # Extraction phase
            mock_resp.choices = [Mock(message=Mock(content='1. Test claim'))]
        elif 'skepticism' in system_prompt or 'check' in system_prompt:
            # Verification phase
            mock_resp.choices = [Mock(message=Mock(content='VERIFIED: True'))]
        elif 'strict editor' in system_prompt:
            # Correction phase
            mock_resp.choices = [Mock(message=Mock(content='Corrected response'))]
        else:
            mock_resp.choices = [Mock(message=Mock(content='Default'))]
        
        mock_resp.usage = Mock(prompt_tokens=10, completion_tokens=20)
        client.calls.append({'model': model, 'phase': call_count[0], 'system': system_prompt[:30]})
        return mock_resp
    
    client.chat.completions.create = mock_create_full
    
    result = hallu.generate(model='default-model', prompt='Test prompt')
    
    # Check that per-phase models were used
    models_used = [call['model'] for call in client.calls]
    print('Models used:', models_used)
    print('Calls:', client.calls)
    
    # The first call (drafting) should use model-draft
    # The second call (extraction) should use model-extract
    # The third call (verification) should use model-verify
    # The fourth call (correction) should use model-correct
    
    assert 'model-draft' in models_used, 'Drafting model not used'
    assert 'model-extract' in models_used, 'Extraction model not used'
    assert 'model-verify' in models_used, 'Verification model not used'
    assert 'model-correct' in models_used, 'Correction model not used'

def test_multi_model_override():
    '''Test that generate() overrides instance defaults.'''
    client = MockClient()
    hallu = AntiHallucinator(
        client=client,
        strictness=1.0,
        model_draft='instance-draft',
        model_extract='instance-extract',
        model_verify='instance-verify',
        model_correct='instance-correct',
    )
    
    call_count = [0]
    def mock_create(model, messages, **kwargs):
        call_count[0] += 1
        system_prompt = messages[0]['content'] if messages else ''
        mock_resp = Mock()
        if 'helpful assistant' in system_prompt:
            mock_resp.choices = [Mock(message=Mock(content='Draft'))]
        elif 'Extract' in system_prompt:
            mock_resp.choices = [Mock(message=Mock(content='1. Claim'))]
        elif 'skepticism' in system_prompt or 'check' in system_prompt:
            mock_resp.choices = [Mock(message=Mock(content='VERIFIED: True'))]
        elif 'strict editor' in system_prompt:
            mock_resp.choices = [Mock(message=Mock(content='Corrected'))]
        mock_resp.usage = Mock(prompt_tokens=10, completion_tokens=20)
        client.calls.append({'model': model, 'phase': call_count[0]})
        return mock_resp
    
    client.chat.completions.create = mock_create
    
    # Override at generate() call time
    result = hallu.generate(
        model='default-model',
        prompt='Test',
        model_draft='override-draft',
        model_extract='override-extract',
        model_verify='override-verify',
        model_correct='override-correct',
    )
    
    models_used = [call['model'] for call in client.calls]
    print('Override models used:', models_used)
    
    assert 'override-draft' in models_used
    assert 'override-extract' in models_used
    assert 'override-verify' in models_used
    assert 'override-correct' in models_used

def test_fallback_to_default_model():
    '''Test that unspecified phases fall back to the default model.'''
    client = MockClient()
    hallu = AntiHallucinator(
        client=client,
        strictness=1.0,
        # Only specify some models
        model_draft='custom-draft',
        model_verify='custom-verify',
    )
    
    call_count = [0]
    def mock_create(model, messages, **kwargs):
        call_count[0] += 1
        system_prompt = messages[0]['content'] if messages else ''
        mock_resp = Mock()
        if 'helpful assistant' in system_prompt:
            mock_resp.choices = [Mock(message=Mock(content='Draft'))]
        elif 'Extract' in system_prompt:
            mock_resp.choices = [Mock(message=Mock(content='1. Claim'))]
        elif 'skepticism' in system_prompt or 'check' in system_prompt:
            mock_resp.choices = [Mock(message=Mock(content='VERIFIED: True'))]
        elif 'strict editor' in system_prompt:
            mock_resp.choices = [Mock(message=Mock(content='Corrected'))]
        mock_resp.usage = Mock(prompt_tokens=10, completion_tokens=20)
        client.calls.append({'model': model, 'phase': call_count[0]})
        return mock_resp
    
    client.chat.completions.create = mock_create
    
    result = hallu.generate(model='default-model', prompt='Test')
    
    models_used = [call['model'] for call in client.calls]
    print('Fallback models used:', models_used)
    
    # Drafting should use custom-draft
    # Extraction should fall back to default-model
    # Verification should use custom-verify
    # Correction should fall back to default-model
    assert 'custom-draft' in models_used
    assert 'custom-verify' in models_used
    assert 'default-model' in models_used

if __name__ == '__main__':
    test_multi_model_routing()
    test_multi_model_override()
    test_fallback_to_default_model()
    print('All tests passed!')
