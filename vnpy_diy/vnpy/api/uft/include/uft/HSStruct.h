/**
* @author �ٺط�
* @company �������ӹɷ����޹�˾
* @file HSStruct.h
* @brief �����˽ӿ�ʹ�õ�ҵ�����ݽṹ
* @date    20170927
*/
#ifndef _HS_API_STRUCT_H_
#define _HS_API_STRUCT_H_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "./HSDataType.h"
#pragma pack(push,4)
struct CHSRspInfoField
{
    /// �������
    HSErrorID					  ErrorID;
    /// ������ʾ
    HSErrorMsg					  ErrorMsg;
};


///������֤
struct CHSReqAuthenticateField
{
	/// �˺�
	HSAccountID                   AccountID;
	/// ����
	HSPassword                    Password;
	/// �ͻ���id
	HSAppID                       AppID;
	/// ��֤��
	HSAuthCode                    AuthCode;
};

///������֤Ӧ��
struct CHSRspAuthenticateField
{
	/// �˺�
	HSAccountID                   AccountID;
	HSAppID                       AppID;
	HSAuthCode                    AuthCode;
};

///�ͻ���¼
struct CHSReqUserLoginField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ����
    HSPassword                    Password;
    /// Ͷ���߶�Ӧ�����
    HSUserApplicationType         UserApplicationType;
    /// Ͷ���߶�Ӧ����Ϣ
    HSUserApplicationInfo         UserApplicationInfo;
    /// Ͷ����Mac��ַ
    HSMacAddress                  MacAddress;
    /// Ͷ����IP��ַ
    HSIPAddress                   IPAddress;
};
///�ͻ���¼Ӧ��
struct CHSRspUserLoginField
{
    /// Ӫҵ����
    HSNum                         BranchID;
    /// �˺�
    HSAccountID                   AccountID;
    /// Ͷ��������
    HSUserName                    UserName;
    /// ������
    HSDate                        TradingDay;
    /// �ϸ�������
    HSDate                        PreTradingDay;
    /// �˵�ȷ�ϱ�־
    HSBillConfirmFlag             BillConfirmFlag;
    /// �Ự����
    HSSessionID                   SessionID;
    /// Ͷ���߶�Ӧ�����
    HSUserApplicationType         UserApplicationType;
    /// Ͷ���߶�Ӧ����Ϣ
    HSUserApplicationInfo         UserApplicationInfo;
    /// ���յȼ�
    HSRiskLevel                   RiskLevel;
    /// Ͷ�����ϴε�½��Mac��ַ
    HSMacAddress                  LastMacAddress;
    /// Ͷ�����ϴε�½��IP��ַ
    HSIPAddress                   LastIPAddress;
    /// �ϴε�¼�ɹ�ʱ��
    HSTime                        LastLoginTime;
    /// ֣������ǰʱ��
    HSTime                        CZCETime;
    /// ��������ǰʱ��
    HSTime                        DCETime;
    /// ��������ǰʱ��
    HSTime                        SHFETime;
    /// �н�����ǰʱ��
    HSTime                        CFFEXTime;
    /// ��Դ���ĵ�ǰʱ��
    HSTime                        INETime;
};

///�������
struct CHSReqUserPasswordUpdateField
{
    /// ��������
    HSPasswordType                PasswordType;
    /// ����
    HSPassword                    Password;
    /// ������
    HSPassword                    NewPassword;
};
///�������Ӧ��
struct CHSRspUserPasswordUpdateField
{
};

///����¼��
struct CHSReqOrderInsertField
{
    /// ��������
    HSRef                         OrderRef;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��������
    HSDirection                   Direction;
    /// ��ƽ��־
    HSOffsetFlag                  OffsetFlag;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// �����۸�
    HSPrice                       OrderPrice;
    /// ��������
    HSVolume                      OrderVolume;
    /// ����ָ��
    HSOrderCommand                OrderCommand;
    /// ��С�ɽ���
    HSVolume                      MinVolume;
    /// ֹ��ֹӯ��
    HSPrice                       SpringPrice;
    /// ������־
    HSSwapOrderFlag               SwapOrderFlag;
};
///����¼��Ӧ��
struct CHSRspOrderInsertField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ��������
    HSOrderSysID                  OrderSysID;
    /// ���͹�˾��������
    HSBrokerOrderID               BrokerOrderID;
    /// �Ự����
    HSSessionID                   SessionID;
    /// ��������
    HSRef                         OrderRef;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��������
    HSDirection                   Direction;
    /// ��ƽ��־
    HSOffsetFlag                  OffsetFlag;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// �����۸�
    HSPrice                       OrderPrice;
    /// ��������
    HSVolume                      OrderVolume;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
    /// ����ָ��
    HSOrderCommand                OrderCommand;
    /// ����ʱ��
    HSTime                        InsertTime;
    /// ��С�ɽ���
    HSVolume                      MinVolume;
    /// ֹ��ֹӯ��
    HSPrice                       SpringPrice;
    /// ������־
    HSSwapOrderFlag               SwapOrderFlag;
};

///����
struct CHSReqOrderActionField
{
    /// ��������
    HSOrderSysID                  OrderSysID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// �Ự����
    HSSessionID                   SessionID;
    /// ��������
    HSRef                         OrderRef;
    /// ������������
    HSRef                         OrderActionRef;
};
///����Ӧ��
struct CHSRspOrderActionField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ��������
    HSOrderSysID                  OrderSysID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// �Ự����
    HSSessionID                   SessionID;
    /// ��������
    HSRef                         OrderRef;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
    /// ����ʱ��
    HSTime                        InsertTime;
    /// ������������
    HSRef                         OrderActionRef;
};

///��Ȩ¼��
struct CHSReqExerciseOrderInsertField
{
    /// ��Ȩ����
    HSRef                         ExerciseRef;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��������
    HSVolume                      OrderVolume;
    /// ��Ȩ��ʽ
    HSExerciseType                ExerciseType;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
};
///��Ȩ¼��Ӧ��
struct CHSRspExerciseOrderInsertField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// �Ự����
    HSSessionID                   SessionID;
    /// ��Ȩ��������
    HSOrderSysID                  ExerciseOrderSysID;
    /// ��Ȩ����
    HSRef                         ExerciseRef;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��������
    HSVolume                      OrderVolume;
    /// ��Ȩ��ʽ
    HSExerciseType                ExerciseType;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
    /// ����ʱ��
    HSTime                        InsertTime;
};

///��Ȩ����
struct CHSReqExerciseOrderActionField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ȩ��������
    HSOrderSysID                  ExerciseOrderSysID;
    /// ��Ȩ����
    HSRef                         ExerciseRef;
    /// �Ự����
    HSSessionID                   SessionID;
    /// ��Ȩ��������
    HSRef                         ExecOrderActionRef;
};
///��Ȩ����Ӧ��
struct CHSRspExerciseOrderActionField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ȩ��������
    HSOrderSysID                  ExerciseOrderSysID;
    /// ��Ȩ����
    HSRef                         ExerciseRef;
    /// �Ự����
    HSSessionID                   SessionID;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
    /// ��Ȩ��������
    HSRef                         ExecOrderActionRef;
};

///����¼��
struct CHSReqLockInsertField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ��������
    HSLockType                    LockType;
    /// ��������
    HSVolume                      OrderVolume;
};
///����¼��Ӧ��
struct CHSRspLockInsertField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ������������
    HSOrderSysID                  LockOrderSysID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ��������
    HSLockType                    LockType;
    /// ��������
    HSVolume                      OrderVolume;
    /// ����ｻ���˺�
    HSAccountID                   UnderlyingAccountID;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
    /// ����ʱ��
    HSTime                        InsertTime;
};

///ѯ��¼��
struct CHSReqForQuoteInsertField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
};
///ѯ��¼��Ӧ��
struct CHSRspForQuoteInsertField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
};

///�������¼��
struct CHSReqCombActionInsertField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��ϲ��Ա���
    HSCombStrategyID              CombStrategyID;
    /// ��ϳֱֲ���
    HSCombPositionID              CombPositionID;
    /// ���ָ���
    HSCombDirection               CombDirection;
    /// ��������
    HSVolume                      OrderVolume;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// ��������
    HSDirection                   Direction;
};
///�������¼��Ӧ��
struct CHSRspCombActionInsertField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ��ϱ�������
    HSOrderSysID                  CombOrderSysID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��ϲ��Ա���
    HSCombStrategyID              CombStrategyID;
    /// ��ϳֱֲ���
    HSCombPositionID              CombPositionID;
    /// ���ָ���
    HSCombDirection               CombDirection;
    /// ��������
    HSVolume                      OrderVolume;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
    /// ����ʱ��
    HSTime                        InsertTime;
    /// ��������
    HSDirection                   Direction;
};

///��󱨵�������ȡ
struct CHSReqQueryMaxOrderVolumeField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ����ָ��
    HSOrderCommand                OrderCommand;
    /// ��������
    HSDirection                   Direction;
    /// ��ƽ��־
    HSOffsetFlag                  OffsetFlag;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// �����۸�
    HSPrice                       OrderPrice;
};
///��󱨵�������ȡӦ��
struct CHSRspQueryMaxOrderVolumeField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ��󱨵���
    HSVolume                      MaxOrderVolume;
    /// ������󱨵���
    HSVolume                      MaxOrderVolumeUnit;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ����ָ��
    HSOrderCommand                OrderCommand;
    /// ��������
    HSDirection                   Direction;
    /// ��ƽ��־
    HSOffsetFlag                  OffsetFlag;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// �����۸�
    HSPrice                       OrderPrice;
};

///������������ȡ
struct CHSReqQueryLockVolumeField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ��������
    HSLockType                    LockType;
};
///������������ȡӦ��
struct CHSRspQueryLockVolumeField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ��������
    HSLockType                    LockType;
    /// ���óֲ�����
    HSVolume                      AvailablePositionVolume;
};

///����Ȩ������ȡ
struct CHSReqQueryExerciseVolumeField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
};
///����Ȩ������ȡӦ��
struct CHSRspQueryExerciseVolumeField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ��󱨵���
    HSVolume                      MaxOrderVolumeUnit;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
};

///�ֲֻ��ܲ�ѯ
struct CHSReqQryPositionField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
};
///�ֲֻ��ܲ�ѯӦ��
struct CHSRspQryPositionField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��������
    HSDirection                   Direction;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// ���ճֲ�����
    HSVolume                      YdPositionVolume;
    /// �ֲ�����
    HSVolume                      PositionVolume;
    /// ���ճֲ�����
    HSVolume                      TodayPositionVolume;
    /// ���óֲ�����
    HSVolume                      AvailablePositionVolume;
    /// ���տ��óֲ�����
    HSVolume                      TodayAvailablePositionVolume;
    /// �ֱֲ�֤��
    HSBalance                     PositionMargin;
    /// Ȩ����
    HSBalance                     Premium;
    /// ������
    HSBalance                     Commission;
    /// ���ֶ�������
    HSVolume                      OpenFrozenVolume;
    /// ƽ�ֶ�������
    HSVolume                      CloseFrozenVolume;
    /// �������
    HSVolume                      CombVolume;
    /// ��Ȩ��������
    HSVolume                      ExerciseFrozenVolume;
    /// ���ᱣ֤��
    HSBalance                     FrozenMargin;
    /// ��Ȩ���ᱣ֤��
    HSBalance                     ExerciseFrozenMargin;
    /// ����Ȩ����
    HSBalance                     FrozenPremium;
    /// ����������
    HSBalance                     FrozenCommission;
    /// ������
    HSVolume                      OpenVolume;
    /// ƽ����
    HSVolume                      CloseVolume;
    /// ���ֽ��
    HSBalance                     OpenBalance;
    /// ƽ�ֽ��
    HSBalance                     CloseBalance;
    /// ���ֳɱ�
    HSBalance                     OpenCost;
    /// �ֲֳɱ�
    HSBalance                     PositionCost;
    /// �ֲ�ӯ��
    HSBalance                     PositionProfit;
    /// ƽ��ӯ��
    HSBalance                     CloseProfit;
    /// ��Ȩ��ֵ
    HSBalance                     OptionsMarketValue;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
};

///�ʽ��˻���ѯ
struct CHSReqQryTradingAccountField
{
};
///�ʽ��˻���ѯӦ��
struct CHSRspQryTradingAccountField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// �������
    HSBalance                     YdBalance;
    /// ���ճֱֲ�֤��
    HSBalance                     YdPositionMargin;
    /// �����ʽ�Ȩ��
    HSBalance                     YdFundEquity;
    /// �ʽ�Ȩ��
    HSBalance                     FundEquity;
    /// ��Ȩ��ֵ
    HSBalance                     OptionsMarketValue;
    /// ��Ȩ��
    HSBalance                     Equity;
    /// �����ʽ�
    HSBalance                     AvailableBalance;
    /// ��ȡ�ʽ�
    HSBalance                     WithdrawBalance;
    /// ��֤��
    HSBalance                     Margin;
    /// ���ᱣ֤��
    HSBalance                     FrozenMargin;
    /// ��Ȩ���ᱣ֤��
    HSBalance                     ExerciseFrozenMargin;
    /// ���ն�
    HSRiskDegree                  RiskDegree;
    /// Ȩ����
    HSBalance                     Premium;
    /// ����Ȩ����
    HSBalance                     FrozenPremium;
    /// ������
    HSBalance                     Commission;
    /// ����������
    HSBalance                     FrozenCommission;
    /// ƽ��ӯ��
    HSBalance                     CloseProfit;
    /// �ֲ�ӯ��
    HSBalance                     PositionProfit;
    /// ƽ�ֶ���ӯ��
    HSBalance                     CloseProfitByDate;
    /// �ֲֶ���ӯ��
    HSBalance                     PositionProfitByDate;
    /// ת����
    HSBalance                     Deposit;
    /// ת�����
    HSBalance                     Withdraw;
    /// ������Ѻ���
    HSBalance                     FundMortgage;
    /// �ֵ���Ѻ���
    HSBalance                     WarrantMortgage;
    /// �����ʽ�
    HSBalance                     FrozenBalance;
    /// �ⶳ�ʽ�
    HSBalance                     UnFrozenBalance;
    /// ����
    HSCurrencyID                  CurrencyID;
};

///������ѯ
struct CHSReqQryOrderField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��������
    HSOrderSysID                  OrderSysID;
};
///������Ϣ
struct CHSOrderField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ��������
    HSOrderSysID                  OrderSysID;
    /// ���͹�˾��������
    HSBrokerOrderID               BrokerOrderID;
    /// �Ự����
    HSSessionID                   SessionID;
    /// ��������
    HSRef                         OrderRef;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��������
    HSDirection                   Direction;
    /// ��ƽ��־
    HSOffsetFlag                  OffsetFlag;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// �����۸�
    HSPrice                       OrderPrice;
    /// ��������
    HSVolume                      OrderVolume;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
    /// �ɽ�����
    HSVolume                      TradeVolume;
    /// ��������
    HSVolume                      CancelVolume;
    /// �ɽ��۸�
    HSPrice                       TradePrice;
    /// ������
    HSDate                        TradingDay;
    /// ��������
    HSDate                        InsertDate;
    /// ����ʱ��
    HSTime                        InsertTime;
    /// �걨ʱ��
    HSTime                        ReportTime;
    /// ����ָ��
    HSOrderCommand                OrderCommand;
    /// ��С�ɽ���
    HSVolume                      MinVolume;
    /// ֹ��ֹӯ��
    HSPrice                       SpringPrice;
    /// ������־
    HSSwapOrderFlag               SwapOrderFlag;
    /// ǿƽԭ��
    HSForceCloseReason            ForceCloseReason;
    /// ������Ϣ
    HSErrorMsg                    ErrorMsg;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ��������
    HSOrderSource                 OrderSource;
};

///�ɽ���ѯ
struct CHSReqQryTradeField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
};
///�ɽ���Ϣ
struct CHSTradeField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// �ɽ�����
    HSTradeID                     TradeID;
    /// ��������
    HSOrderSysID                  OrderSysID;
    /// ���͹�˾��������
    HSBrokerOrderID               BrokerOrderID;
    /// �Ự����
    HSSessionID                   SessionID;
    /// ��������
    HSRef                         OrderRef;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��������
    HSDirection                   Direction;
    /// ��ƽ��־
    HSOffsetFlag                  OffsetFlag;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// �ɽ�����
    HSVolume                      TradeVolume;
    /// �ɽ��۸�
    HSPrice                       TradePrice;
    /// ������
    HSDate                        TradingDay;
    /// �ɽ�ʱ��
    HSTime                        TradeTime;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
};

///��Ȩ��ѯ
struct CHSReqQryExerciseField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��Ȩ��������
    HSOrderSysID                  ExerciseOrderSysID;
};
///��Ȩ��Ϣ
struct CHSExerciseField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ��Ȩ��������
    HSOrderSysID                  ExerciseOrderSysID;
    /// �Ự����
    HSSessionID                   SessionID;
    /// ��Ȩ����
    HSRef                         ExerciseRef;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// ��Ȩ��ʽ
    HSExerciseType                ExerciseType;
    /// ��Ȩ����
    HSVolume                      OrderVolume;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
    /// ��������
    HSDate                        InsertDate;
    /// ����ʱ��
    HSTime                        InsertTime;
    /// ������Ϣ
    HSErrorMsg                    ErrorMsg;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ��������
    HSOrderSource                 OrderSource;
    /// ��Ȩ��Ȩ�����ɵ�ͷ���Ƿ��Զ�ƽ��
    HSCloseFlag                   CloseFlag;
    /// ��ƽ��־
    HSOffsetFlag                  OffsetFlag;
};

///������ѯ
struct CHSReqQryLockField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ������������
    HSOrderSysID                  LockOrderSysID;
};
///������Ϣ
struct CHSLockField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ������������
    HSOrderSysID                  LockOrderSysID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ��������
    HSLockType                    LockType;
    /// ��������
    HSVolume                      OrderVolume;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
    /// ������
    HSDate                        TradingDay;
    /// ����ʱ��
    HSTime                        InsertTime;
    /// ������Ϣ
    HSErrorMsg                    ErrorMsg;
    /// ��������
    HSOrderSource                 OrderSource;
};

///������ϲ�ѯ
struct CHSReqQryCombActionField
{
    /// ��ϱ�������
    HSOrderSysID                  CombOrderSysID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
};
///���������Ϣ
struct CHSCombActionField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ��ϱ�������
    HSOrderSysID                  CombOrderSysID;
    /// ��ϳֱֲ���
    HSCombPositionID              CombPositionID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��ϲ��Ա���
    HSCombStrategyID              CombStrategyID;
    /// ��������
    HSDirection                   Direction;
    /// ���ָ���
    HSCombDirection               CombDirection;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// ��������
    HSVolume                      OrderVolume;
    /// ����״̬
    HSOrderStatus                 OrderStatus;
    /// ������
    HSDate                        TradingDay;
    /// ��������
    HSDate                        InsertDate;
    /// ����ʱ��
    HSTime                        InsertTime;
    /// ������Ϣ
    HSErrorMsg                    ErrorMsg;
    /// ��������
    HSOrderSource                 OrderSource;
};

///��ϳֲ���ϸ��ѯ
struct CHSReqQryPositionCombineDetailField
{
};
///��ϳֲ���ϸ��ѯӦ��
struct CHSRspQryPositionCombineDetailField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ��ϳֱֲ���
    HSCombPositionID              CombPositionID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��ϲ��Ա���
    HSCombStrategyID              CombStrategyID;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// ��������
    HSDirection                   Direction;
    /// ���ճֲ�����
    HSVolume                      YdPositionVolume;
    /// ���óֲ�����
    HSVolume                      AvailablePositionVolume;
    /// ��ϻ�������
    HSVolume                      TotalCombVolume;
    /// ��ֻ�������
    HSVolume                      TotalSplitVolume;
    /// �ֱֲ�֤��
    HSBalance                     PositionMargin;
    /// ��������
    HSDate                        OpenDate;
};

///��Լ��Ϣ��ѯ
struct CHSReqQryInstrumentField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
};
///��Լ��Ϣ��ѯӦ��
struct CHSRspQryInstrumentField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��Լ����
    HSInstrumentName              InstrumentName;
    /// ��ԼӢ������
    HSInstrumentEngName           InstrumentEngName;
    /// ��ԼƷ�ֱ���
    HSProductID                   ProductID;
    /// ��Ʒ����
    HSProductType                 ProductType;
    /// �м۵���󱨵���
    HSVolume                      MaxMarketOrderVolume;
    /// �м۵���С������
    HSVolume                      MinMarketOrderVolume;
    /// �޼۵���󱨵���
    HSVolume                      MaxLimitOrderVolume;
    /// �޼۵���С������
    HSVolume                      MinLimitOrderVolume;
    /// ��Լ��������
    HSVolume                      VolumeMultiple;
    /// ��С�䶯��λ
    HSPrice                       PriceTick;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ��Ȩ��
    HSPrice                       ExercisePrice;
    /// ��Ȩ����
    HSOptionsType                 OptionsType;
    /// ���ױ�־
    HSTradingFlag                 TradingFlag;
    /// ������
    HSDate                        MarketDate;
    /// ������
    HSDate                        ExpireDate;
    /// ��Ȩ��ʼ����
    HSDate                        BeginExerciseDate;
    /// ��Ȩ��������
    HSDate                        EndExerciseDate;
    /// ��ʼ������
    HSDate                        BeginDeliveryDate;
    /// ����������
    HSDate                        EndDeliveryDate;
    /// ����
    HSCurrencyID                  CurrencyID;
    /// �������
    HSCombType                    CombType;
};

///����ȱ�ڲ�ѯ
struct CHSReqQryCoveredShortField
{
    /// ����������
    HSExchangeID                  ExchangeID;
};
///����ȱ�ڲ�ѯӦ��
struct CHSRspQryCoveredShortField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ������������
    HSVolume                      CoveredLockVolume;
    /// ����ȱ������
    HSVolume                      CoveredShortVolume;
    /// ����Ԥ��ȱ������
    HSVolume                      CoveredEstimateShortVolume;
    /// ��Ϣ����
    HSMsgContent                   MsgContent;
};

///��Ȩָ�ɲ�ѯ
struct CHSReqQryExerciseAssignField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// �ֲ�����
    HSPositionType                PositionType;
};
///��Ȩָ�ɲ�ѯӦ��
struct CHSRspQryExerciseAssignField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// �ֲ�����
    HSPositionType                PositionType;
    /// ��Ȩ����
    HSOptionsType                 OptionsType;
    /// ��Ȩ��
    HSPrice                       ExercisePrice;
    /// ��Ȩ����
    HSVolume                      ExerciseVolume;
    /// ��������
    HSVolume                      DeliveryVolume;
    /// ��Ȩ�����ʽ�
    HSBalance                     ExerciseFrozenBalance;
    /// ������
    HSBalance                     SettlementBalance;
};

///����ת��
struct CHSReqTransferField
{
    /// ���д���
    HSBankID                      BankID;
    /// ת������
    HSTransferType                TransferType;
    /// �������
    HSBalance                     OccurBalance;
    /// �ʽ�����
    HSPassword                    FundPassword;
    /// ��������
    HSPassword                    BankPassword;
    /// ����
    HSCurrencyID                  CurrencyID;
    /// ת�˳���
    HSOccasion                    TransferOccasion;
};
///����ת��Ӧ��
struct CHSRspTransferField
{
    /// ת����ˮ
    HSNum                         TransferSerialID;
    /// ���д���
    HSBankID                      BankID;
    /// ת������
    HSTransferType                TransferType;
    /// �������
    HSBalance                     OccurBalance;
    /// ����
    HSCurrencyID                  CurrencyID;
    /// ת�˳���
    HSOccasion                    TransferOccasion;
};

///����ת�˲�ѯ
struct CHSReqQryTransferField
{
    /// ���д���
    HSBankID                      BankID;
    /// ת����ˮ
    HSNum                         TransferSerialID;
};
///����ת�˲�ѯӦ��
struct CHSRspQryTransferField
{
    /// ת����ˮ
    HSNum                         TransferSerialID;
    /// ���д���
    HSBankID                      BankID;
    /// ��������
    HSBankName                    BankName;
    /// ҵ������
    HSBusinessName                BusinessName;
    /// �������
    HSBalance                     OccurBalance;
    /// ���ʽ��
    HSBalance                     PostBalance;
    /// ת��ʱ��
    HSTime                        TransferTime;
    /// ת��״̬
    HSTransferStatus              TransferStatus;
    /// ת�˷���
    HSTransferSource              TransferSource;
    /// ��ע
    HSRemarks                     Remarks;
    /// ����
    HSCurrencyID                  CurrencyID;
    /// ���׷�������
    HSDate                        OrderSourceDate;
    /// ��������
    HSDate                        TradingDay;
    /// ת�˳���
    HSOccasion                    TransferOccasion;
};

///��������ѯ
struct CHSReqQueryBankBalanceField
{
    /// ���д���
    HSBankID                      BankID;
    /// �ʽ�����
    HSPassword                    FundPassword;
    /// ��������
    HSPassword                    BankPassword;
    /// ����
    HSCurrencyID                  CurrencyID;
};
///��������ѯӦ��
struct CHSRspQueryBankBalanceField
{
    /// ת����ˮ
    HSNum                         TransferSerialID;
};

///�����˻���ѯ
struct CHSReqQueryBankAccountField
{
    /// ���д���
    HSBankID                      BankID;
    /// ����
    HSCurrencyID                  CurrencyID;
};
///�����˻���ѯӦ��
struct CHSRspQueryBankAccountField
{
    /// ���д���
    HSBankID                      BankID;
    /// ��������
    HSBankName                    BankName;
    /// �����˺�
    HSBankAccountID               BankAccountID;
    /// ����
    HSCurrencyID                  CurrencyID;
};

///�ͻ��˵���ѯ
struct CHSReqQueryBillContentField
{
    /// ��ʼ����
    HSDate                        BeginDate;
    /// ��������
    HSDate                        EndDate;
};
///�ͻ��˵���ѯӦ��
struct CHSRspQueryBillContentField
{
    /// �˵�����
    HSBillContent                 BillContent;
};

///�ͻ��˵�ȷ��
struct CHSReqBillConfirmField
{
};
///�ͻ��˵�ȷ��Ӧ��
struct CHSRspBillConfirmField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// �˵�ȷ��״̬
    HSBillConfirmStatus           BillConfirmStatus;
    /// ȷ������
    HSDate                        ConfirmDate;
    /// ȷ��ʱ��
    HSTime                        ConfirmTime;
};

///��֤���ѯ
struct CHSReqQryMarginField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
};
///��֤���ѯӦ��
struct CHSRspQryMarginField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// ��ͷ��֤����
    HSRate                        LongMarginRatio;
    /// ��ͷ��֤��
    HSBalance                     LongMargin;
    /// ��ͷ��֤����
    HSRate                        ShortMarginRatio;
    /// ��ͷ��֤��
    HSBalance                     ShortMargin;
};

///�����Ѳ�ѯ
struct CHSReqQryCommissionField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ʒ����
    HSProductType                 ProductType;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
};
///�����Ѳ�ѯӦ��
struct CHSRspQryCommissionField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Ʒ����
    HSProductType                 ProductType;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��Ȩ��Ӧ�ı�ĺ�Լ����
    HSInstrumentID                UnderlyingInstrID;
    /// ��ͷ������������
    HSRate                        LongOpenCommissionRatio;
    /// ��ͷ����������
    HSBalance                     LongOpenCommission;
    /// ��ͷƽ����������
    HSRate                        LongCloseCommissionRatio;
    /// ��ͷƽ��������
    HSBalance                     LongCloseCommission;
    /// ��ͷƽ����������
    HSRate                        LongCloseTodayCommissionRatio;
    /// ��ͷƽ��������
    HSBalance                     LongCloseTodayCommission;
    /// ��ͷ������������
    HSRate                        ShortOpenCommissionRatio;
    /// ��ͷ����������
    HSBalance                     ShortOpenCommission;
    /// ��ͷƽ����������
    HSRate                        ShortCloseCommissionRatio;
    /// ��ͷƽ��������
    HSBalance                     ShortCloseCommission;
    /// ��ͷƽ����������
    HSRate                        ShortCloseTodayCommissionRatio;
    /// ��ͷƽ��������
    HSBalance                     ShortCloseTodayCommission;
    /// ��Ȩ��������
    HSRate                        ExerciseCommissionRatio;
    /// ��Ȩ������
    HSBalance                     ExerciseCommission;
    /// �ɽ���ֵ����
    HSRate                        TradeValueRatio;
    /// ��Ʊ��ֵ
    HSBalance                     StockValue;
};

///���ʲ�ѯ
struct CHSReqQryExchangeRateField
{
	/// ԭ����
	HSCurrencyID                  FromCurrencyID;
	/// Ŀ�����
	HSCurrencyID                  ToCurrencyID;
};
///���ʲ�ѯӦ��
struct CHSRspQryExchangeRateField
{
	/// ԭ����
	HSCurrencyID                  FromCurrencyID;
	/// Ŀ�����
	HSCurrencyID                  ToCurrencyID;
	/// ����
	HSExchangeRate                ExchangeRate;
	/// ԭ����������λ
	HSVolume                      FromCurrencyUnit;
};

///�ֲ���ϸ��ѯ
struct CHSReqQryPositionDetailField
{
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
};
///�ֲ���ϸ��ѯӦ��
struct CHSRspQryPositionDetailField
{
    /// �˺�
    HSAccountID                   AccountID;
    /// �ɽ�����
    HSTradeID                     TradeID;
    /// ��������
    HSDate                        OpenDate;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ��������
    HSDirection                   Direction;
    /// Ͷ��/�ױ�/��������
    HSHedgeType                   HedgeType;
    /// �ɽ�����
    HSVolume                      TradeVolume;
    /// �ֲּ۸�
    HSPrice                       PositionPrice;
    /// ����ƽ������
    HSVolume                      TodayCloseVolume;
    /// �����ֲֺ�
    HSArbitPositionID             ArbitragePositionID;
    /// �Ⱥ�
    HSLegID                       LegID;
    /// �ɽ�����
    HSTradeType                   TradeType;
    /// �ֱֲ�֤��
    HSBalance                     PositionMargin;
    /// ��Ϻ�Լ����
    HSInstrumentID                CombInstrumentID;
};

///���͹�˾���ò�����ѯ
struct CHSReqQrySysConfigField
{
};
///���͹�˾���ò�����ѯӦ��
struct CHSRspQrySysConfigField
{
    /// ���ñ��
    HSConfigNo                    ConfigNo;
    /// ���ÿ���״̬
    HSConfigValue                 ConfigValue;
};

/// ���鶩�ģ�ȡ����������
struct CHSReqDepthMarketDataField
{
    HSExchangeID                  ExchangeID;
    HSInstrumentID                InstrumentID;
};

///�����ѯ
struct CHSReqQryDepthMarketDataField
{
	/// ����������
	HSExchangeID                  ExchangeID;
	/// ��Լ����
	HSInstrumentID                InstrumentID;
};

///������Ϣ
struct CHSDepthMarketDataField
{
    /// ������
    HSDate                        TradingDay;
    /// ��Լ����
    HSInstrumentID                InstrumentID;
    /// ����������
    HSExchangeID                  ExchangeID;
    /// ���¼�
    HSPrice                       LastPrice;
    /// ������
    HSPrice                       PreSettlementPrice;
    /// �����̼�
    HSPrice                       PreClosePrice;
    /// ���̼�
    HSPrice                       OpenPrice;
    /// ��߼�
    HSPrice                       HighestPrice;
    /// ��ͼ�
    HSPrice                       LowestPrice;
    /// �ɽ�����
    HSVolume                      TradeVolume;
    /// �ɽ����
    HSBalance                     TradeBalance;
    /// �ܳ���
    HSVolume                      OpenInterest;
    /// ���̼�
    HSPrice                       ClosePrice;
    /// �����
    HSPrice                       SettlementPrice;
    /// ��ͣ���
    HSPrice                       UpperLimitPrice;
    /// ��ͣ���
    HSPrice                       LowerLimitPrice;
    /// ����ʵ��
    HSDelta                       PreDelta;
    /// ����ʵ��
    HSDelta                       CurrDelta;
    /// ������ʱ��
    HSTime                        UpdateTime;
    /// �����һ
    HSPrice                       BidPrice1;
    /// ������һ
    HSVolume                      BidVolume1;
    /// ������һ
    HSPrice                       AskPrice1;
    /// ������һ
    HSVolume                      AskVolume1;
    /// ����۶�
    HSPrice                       BidPrice2;
    /// ��������
    HSVolume                      BidVolume2;
    /// �����۶�
    HSPrice                       AskPrice2;
    /// ��������
    HSVolume                      AskVolume2;
    /// �������
    HSPrice                       BidPrice3;
    /// ��������
    HSVolume                      BidVolume3;
    /// ��������
    HSPrice                       AskPrice3;
    /// ��������
    HSVolume                      AskVolume3;
    /// �������
    HSPrice                       BidPrice4;
    /// ��������
    HSVolume                      BidVolume4;
    /// ��������
    HSPrice                       AskPrice4;
    /// ��������
    HSVolume                      AskVolume4;
    /// �������
    HSPrice                       BidPrice5;
    /// ��������
    HSVolume                      BidVolume5;
    /// ��������
    HSPrice                       AskPrice5;
    /// ��������
    HSVolume                      AskVolume5;
    /// ƽ���۸�
    HSPrice                       AveragePrice;
    /// ��ֲ���
    HSVolume                      PreOpenInterest;
    /// ��Լ����״̬
    HSInstrumentTradeStatus       InstrumentTradeStatus;
    /// ��Լʵʱ��������
    HSOpenRestriction             OpenRestriction;
};


#pragma pack(pop)
#endif
