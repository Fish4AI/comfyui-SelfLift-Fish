"""Native H3/SLA/sol-attn regression checks for bounded same-forward reuse.

Run with ComfyUI's Python and --comfy-root PATH. No model download or service.
"""
import argparse
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

parser=argparse.ArgumentParser();parser.add_argument('--comfy-root',required=True)
args,remaining=parser.parse_known_args();sys.path.insert(0,args.comfy_root);sys.argv=[sys.argv[0]]
import torch
import comfy.ops
from comfy.quant_ops import QuantizedTensor
from comfy.ldm.minimax import model as h3
from comfy_extras.nodes_sparse_attention import SparseAttnPatch

pkg=types.ModuleType('fish_reuse_tests');pkg.__path__=[str(Path(__file__).resolve().parents[1])]
sys.modules[pkg.__name__]=pkg
stream=importlib.import_module(pkg.__name__+'.h3_stream_attention')
global_tiling=importlib.import_module(pkg.__name__+'.h3_global_tiling')
prepared=importlib.import_module(pkg.__name__+'.h3_prepared_weights')

class BudgetTests(unittest.TestCase):
    def test_low_stage_without_references_is_untouched(self):
        options={'original':object()}
        class Executor:
            def __call__(self,streams,timestep,context,opts,**kwargs):
                assert opts is options
                return streams
        streams=[torch.tensor([1.])]
        with patch.object(prepared,'weight_cache_budget',side_effect=AssertionError('must not run')):
            self.assertIs(global_tiling._native_forward_reuse(Executor(),streams,None,None,options,
                base_model_ref=lambda:None,weight_cache_gb=4,minimax_payload={}),streams)

    def test_cap_headroom_disabled_and_cpu(self):
        device=torch.device('cuda')
        with patch.object(torch.cuda,'mem_get_info',return_value=(8*2**30,16*2**30)) as query:
            self.assertEqual(stream._input_cache_budget(device,900),256*2**20)
            self.assertEqual(stream._input_cache_budget(device,0),0)
            self.assertEqual(stream._input_cache_budget(torch.device('cpu'),256),0)
            self.assertEqual(query.call_count,1)
        with patch.object(torch.cuda,'mem_get_info',return_value=(2*2**30+32*2**20,16*2**30)):
            self.assertEqual(stream._input_cache_budget(device,256),32*2**20)
        with patch.object(torch.cuda,'mem_get_info',return_value=(2**30,16*2**30)):
            self.assertEqual(stream._input_cache_budget(device,256),0)

    def test_native_wrapper_preserves_existing_patch_and_restores_on_failure(self):
        events=[]
        @contextmanager
        def budget(*args):
            events.append('budget_enter')
            try:yield
            finally:events.append('budget_exit')
        @contextmanager
        def weights(*args):
            events.append('weights_enter')
            try:yield
            finally:events.append('weights_exit')
        def prior(args,extra):
            self.assertEqual(extra['marker'],91)
            return {'img':args['img']+7}
        options={'patches_replace':{'dit':{('double_block',0):prior}},'marker':12}
        block=object();x=torch.tensor([2.])
        class Executor:
            class_obj=types.SimpleNamespace(blocks=[block])
            def __call__(self,streams,timestep,context,opts,**kwargs):
                assert opts['marker']==12 and kwargs['minimax_payload']=={'refs':['unchanged']}
                return opts['patches_replace']['dit'][('double_block',0)](
                    {'img':streams[0]}, {'marker':91})['img']
        with patch.object(prepared,'weight_cache_budget',budget),patch.object(prepared,'prepared_block',weights):
            result=global_tiling._native_forward_reuse(Executor(),[x],None,None,options,
                base_model_ref=lambda:None,weight_cache_gb=4,minimax_payload={'refs':['unchanged']})
            self.assertEqual(result.item(),9)
            self.assertIs(options['patches_replace']['dit'][('double_block',0)],prior)
            self.assertEqual(events,['budget_enter','weights_enter','weights_exit','budget_exit'])
            events.clear()
            def failure(*args):raise RuntimeError('injected failure')
            options['patches_replace']['dit'][('double_block',0)]=failure
            with self.assertRaisesRegex(RuntimeError,'injected failure'):
                global_tiling._native_forward_reuse(Executor(),[x],None,None,options,
                    base_model_ref=lambda:None,weight_cache_gb=4,minimax_payload={'refs':['unchanged']})
            self.assertEqual(events,['budget_enter','weights_enter','weights_exit','budget_exit'])

@unittest.skipUnless(torch.cuda.is_available(),'Native attention requires CUDA')
class InputReuseTests(unittest.TestCase):
    @torch.inference_mode()
    def test_exact_outputs_pooled_state_partial_chunks_and_no_cross_step_reuse(self):
        torch.manual_seed(82731)
        block=h3.DiTBlock(64,18,128,128,32,1e-5,1e-5,dtype=torch.bfloat16,device='cuda',
                          operations=comfy.ops.manual_cast)
        for param in block.parameters():param.copy_(torch.randn_like(param)*.025)
        block.requires_grad_(False)
        raw_weight=block.attn.qkv_proj.weight.detach().clone()
        # A head tail, token tail, and per-token modulation rows cross chunks.
        n=193
        x=torch.randn(n,64,device='cuda',dtype=torch.bfloat16)
        temb=torch.randn(2,32,device='cuda',dtype=torch.bfloat16)
        segments=[(0,73,1),(73,129,torch.arange(56,device='cuda')%6),(129,n,0)]
        rope=h3.rope_rotation_table(torch.randn(n,96,device='cuda')*.1,torch.bfloat16)
        layout=types.SimpleNamespace(seq_len=n,segments=[(0,32,'text'),(32,96,'ref_img'),(96,128,'audio'),(128,n,'video')])
        options={'minimax_h3_layout':layout,'uuids':['test']}
        for fmt in (None,'TensorCoreFP8E4M3Layout','TensorWiseINT8Layout'):
            weight=raw_weight if fmt is None else QuantizedTensor.from_float(
                raw_weight,fmt,**({'per_channel':True} if fmt=='TensorWiseINT8Layout' else {'scale':'recalculate'}))
            block.attn.qkv_proj.weight=torch.nn.Parameter(weight,requires_grad=False)
            block.attn.qkv_proj.input_scale=torch.tensor(.03125,device='cuda')
            block.attn.qkv_proj.pre_quant_scale=torch.linspace(.8,1.2,64,device='cuda',dtype=torch.bfloat16)
            for ratio in (.1,0.):
                for cap in (0,8192,256*2**20):
                    patches=[SparseAttnPatch(1.3,ratio,False,1.,0.,0,set(),'exact_kv_and_rows',0,False) for _ in range(2)]
                    for step in range(2):
                        outputs=[];counts=[]
                        for i,budget in enumerate((0,cap)):
                            with patch.object(stream,'_input_cache_budget',return_value=budget),\
                                 patch.object(block.norm1,'forward',wraps=block.norm1.forward) as norm:
                                out=stream.sparse_block(block,x.clone()+step*.125,temb,segments,rope,options,64,(patches[i],None),0)
                                outputs.append(out);counts.append(norm.call_count)
                        with self.subTest(format=fmt,ratio=ratio,cap=cap,step=step):
                            torch.testing.assert_close(*outputs,rtol=0,atol=0)
                            for a,b in zip(next(iter(patches[0].pooled.values())),next(iter(patches[1].pooled.values()))):
                                torch.testing.assert_close(a,b,rtol=0,atol=0)
                            if cap==256*2**20:self.assertEqual(counts[1],4)
                            if cap:self.assertLess(counts[1],counts[0])
                            else:self.assertEqual(counts[1],counts[0])

if __name__=='__main__':unittest.main(argv=[sys.argv[0],*remaining])
