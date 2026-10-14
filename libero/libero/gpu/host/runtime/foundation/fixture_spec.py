"""Frozen source-validator fixture identity, independent of a raw record.

Matches original GPU metadata controls and retains their cases, source protocol,
static capacity and original stage IDs. It is not a physics oracle/case subset.
"""
WORLD_CAPACITIES=(1,4,20,128)
CAPACITY=4
FIRST_STAGE=11
REPEAT_STAGE=12
PROTOCOLS=('D04.factor_solve','D04.result_kind','D09.final_status','D04.body_status','D04.frame_status','D07.jac_status','D07.adapter_status','C05.dispatch_status','S01.contact_status','S02.scalar_status','S04.update_status','S05.hessian_status','S06.direction_status','S07.linesearch_status','S08.A_status','S08.B_status','S08.C_status','S08.D_status')
SCENARIOS=('zero_count','inactive_invalid_count','invalid_active','negative_count','capacity_exact','undercapacity','wrong_epoch','invalid_producer_valid','sticky_first_error')
LOCAL_CODES=tuple(range(-1,34))
INITIAL_OUT_BITS=0x7ff8000000000123

def case_identity(case,worlds):
    if type(worlds) is not int or worlds not in WORLD_CAPACITIES:raise AssertionError('unknown frozen world capacity')
    if type(case) is not str:raise AssertionError('invalid frozen fixture label')
    if case in SCENARIOS:return {'protocol':'D09.final_status','code':0,'scenario':case,'N':worlds,'capacity':CAPACITY,'stage':FIRST_STAGE,'path':'raw_N'+str(worlds)+'_'+case+'.npz'}
    if case in ('numeric_signedzero','numeric_nonzero') and worlds==4:
        return {'protocol':'D09.final_status','code':0,'scenario':'inactive_invalid_count','N':worlds,'capacity':CAPACITY,'stage':FIRST_STAGE,'path':'NUMERIC_'+case[len('numeric_'):]+'.npz','initial_bits':0}
    negative={'generic_both_negative':(-1,-1),'generic_producer_negative':(-1,7),'generic_expected_negative':(7,-1),'generic_min_negative':(-2**63,-2**63)}
    if case in negative:
        producer,expected=negative[case]
        return {'protocol':'D09.final_status','code':0,'scenario':'generic_negative','N':worlds,'capacity':CAPACITY,'stage':FIRST_STAGE,'path':'GENERIC_N'+str(worlds)+'_'+case[len('generic_'):]+'.npz','producer_epoch':producer,'expected_epoch':expected,'initial_bits':0xc074100000000000}
    for name in PROTOCOLS:
        for code in LOCAL_CODES:
            if case=='local_'+name.replace('.','_')+'_'+str(code):return {'protocol':name,'code':code,'scenario':'valid','N':worlds,'capacity':CAPACITY,'stage':FIRST_STAGE,'path':'raw_N'+str(worlds)+'_'+case+'.npz'}
    raise AssertionError('fixture label is outside frozen namespace/code domain')

def validate_record(record):
    spec=case_identity(record.get('case'),record.get('N'))
    for key in ('protocol','capacity','stage'):
        if record.get(key)!=spec[key] or (key!='protocol' and type(record.get(key)) is not int):raise AssertionError('record differs from frozen fixture '+key)
    if record['stage']<=0 or record['stage']>2**31-1:raise AssertionError('positive original int32 stage required')
    if 'path' in record and record['path']!=spec['path']:raise AssertionError('raw case path differs from frozen fixture')
    return spec

def input_spec(spec):
    W=spec['N'];s=spec['scenario'];values={'admission':[1]*W,'count':[1]*W,'producer_epoch':[7]*W,'expected_epoch':[7]*W,'producer_valid':[1]*W,'event':[101+w for w in range(W)],'local_status':[spec['code']]*W,'error_item':[17+w for w in range(W)]}
    if s=='generic_negative':values.update(producer_epoch=[spec['producer_epoch']]*W,expected_epoch=[spec['expected_epoch']]*W,event=[111]*W,error_item=[-1]*W)
    if s=='zero_count':values['count']=[0]*W
    if s=='inactive_invalid_count':values.update(admission=[0]*W,count=[-2147483648]*W,producer_valid=[0]*W,producer_epoch=[-1]*W)
    if s=='invalid_active':values['admission']=[2]*W
    if s=='negative_count':values['count']=[-1]*W
    if s=='capacity_exact':values['count']=[CAPACITY]*W
    if s=='undercapacity':values['count']=[CAPACITY+1]*W
    if s=='wrong_epoch':values['producer_epoch']=[6]*W
    if s=='invalid_producer_valid':values['producer_valid']=[0]*W
    if s=='sticky_first_error':values['local_status']=[-1 if w%2==0 else 3 for w in range(W)]
    return values
