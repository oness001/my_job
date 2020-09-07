#ifndef __KSGOLDUSERAPISTRUCTEX__H__
#define __KSGOLDUSERAPISTRUCTEX__H__

#include "KSGoldUserApiDataTypeEx.h"

namespace KSGoldTradeAPI
{

enum KS_TE_RESUME_TYPE
{
	KS_TERT_RESTART = 0,
	KS_TERT_RESUME,
	KS_TERT_QUICK
};

///�û���¼����
struct CThostFtdcReqUserLoginField
{
	TThostFtdcTraderIDType     AccountID;			//��¼�ʺ�
	TThostFtdcLoginType        LoginType;           //��¼�˺�����
	TThostFtdcPasswordType     Password;			//����
	TThostFtdcIPAddressType	   LoginIp;				//�ն�IP��ַ
	TThostFtdcMacAddressType   MacAddress;			//Mac��ַ
	TThostFtdcProductInfoType  UserProductionInfo;	//�û��˲�Ʒ��Ϣ
	TThostFtdcProtocolInfoType ProtocolInfo;		//Э����Ϣ
	//TThostFtdcDiskSerialNoType DiskSerialNo;		//Ӳ�����к�
};

///�û���¼Ӧ��
struct CThostFtdcRspUserLoginField
{
	TThostFtdcDateType				 TradeDate;				//��������
	TThostFtdcLoginBatchType		 SessionID;				//�Ự���
	TThostFtdcClientIDType			 ClientID;         		//�ͻ���
	TThostFtdcCSRCInvestorNameType	 clientName;			//�ͻ�����
	TThostFtdcClientIDType			 TradeCode;			   	//���ױ���	
	TThostFtdcSeatIDType			 SeatNo;				//ϯλ��
	TThostFtdcIPAddressType			 lastLoginIp;			//�ϴε�¼IP��ַ
	TThostFtdcDateType				 lastLoginDate;			//�ϴε�¼����
	TThostFtdcTimeType				 lastLoginTime;			//�ϴε�¼ʱ��
	TThostFtdcTraderIDType			 AccountID;				//��¼�˺�
	TThostFtdcLoginType				 LoginType;				//��¼�˺�����
	TThostFtdcPasswordType			 Password;				//����
	TThostFtdcMacAddressType		 MacAddress;			//MAC��ַ
	TThostFtdcIPAddressType			 LoginIp;				//�ն�IP��ַ
	TThostFtdcProductInfoType		 UserProductionInfo;	//�û��˲�Ʒ��Ϣ
	TThostFtdcProtocolInfoType		 ProtocolInfo;			//Э����Ϣ
	TThostFtdcSystemNameType		 SystemName;			//����ϵͳ����
	TThostFtdcFrontIDType			 FrontID;			    //ǰ�ñ��
	TThostFtdcOrderRefType			 MaxOrderRef;			//��󱨵�����
	TThostFtdcTimeType			 	 SgeTime;			    //����ʱ��
};

///�û��ǳ�����
struct CThostFtdcUserLogoutField
{
	TThostFtdcIPAddressType	        LoginIp;				//�ն�IP��ַ
	TThostFtdcMacAddressType        MacAddress;			    //Mac��ַ
	TThostFtdcClientIDType			ClientID;         		//�ͻ���
};

///�ͻ��Ự��Ϣͨ�ò�������
struct CThostFtdcQryClientSessionField
{
	TThostFtdcClientIDType			 ClientID;         		//�ͻ���
};

///�ͻ��Ự��Ϣͨ�ò���Ӧ��
struct CThostFtdcRspClientSessionField
{
	TThostFtdcClientIDType			 ClientID;         		//�ͻ���
	TThostFtdcLoginBatchType		 SessionID;				//�ỰID
	TThostFtdcIdDiffCode             IdDiffCode;			//�������������
	TThostFtdcIPAddressType			 CurLoginIp;			//��¼IP��ַ
	TThostFtdcMacAddressType		 CurMacAddress;			//MAC��ַ
	TThostFtdcStatus				 Status;				//��¼״̬
	TThostFtdcTimeType				 LoginTime;				//��¼ʱ��
	TThostFtdcDateType				 LoginDate;				//��¼����
	TThostFtdcTimeType				 LogoutTime;			//¼��ʱ��
	TThostFtdcDateType				 LogoutDate;			//¼������
	TThostFtdcFailNumber			 FailNumber;			//��¼ʧ�ܴ���
};

///��ѯ����Ʒ��Ӧ��
struct CThostFtdcRspVarietyCodeField
{
	TThostFtdcExchangeIDType	    ExchangeID;			//����������	
	TThostFtdcAbbrType				VarietyShortName;	//Ʒ�ּ��	
	TThostFtdcAbbrType				VarietyFullName;	//Ʒ��ȫ��		
	TThostFtdcInstrumentIDType		VarietyCode;	    //����Ʒ�ִ���
	TThostFtdcDeliveryVarietyType	VarietyType;        //Ʒ�����
	TThostFtdcWeightUnit			WeightUnit;			//������λ
	TThostFtdcOrderStatusType		Status;				//״̬
	TThostFtdcMinTakeVolumn			MinTakeVolumn;		//��С�����
	TThostFtdcTakeUnit				TakeUnit;			//�����λ		
	TThostFtdcWeightType			Weight;				//Ĭ������		
	TThostFtdcProductIDType			ProductID;			//��Ʒ����	
};

///��ѯ��ԼӦ��
struct CThostFtdcInstrumentField
{
	TThostFtdcExchangeIDType	    ExchangeID;   //����������	
	TThostFtdcInstrumentIDType		InstID;       //��Լ����	
	TThostFtdcRateType				LowerLimit;   //��ͣ����		
	TThostFtdcMarketIDType			MarketID;     //�г�����
	TThostFtdcVolumeType			MaxHand;      //����걨����
	TThostFtdcVolumeType			MinHand;      //��С�걨����
	TThostFtdcInstrumentNameType    Name;         //��Լ����
	TThostFtdcOpenFlagType			OpenFlag;     //��Ծ��־
	TThostFtdcPriceType				Tick;         //��С�䶯��λ		
	TThostFtdcInstStateFlagType		TradeState;	  //��Լ����״̬		
	TThostFtdInstUnitType			Unit;         //���׵�λ����
	TThostFtdcRateType				UpperLimit;   //��ͣ����		
	TThostFtdcVarietyIDType			VarietyID;    //����Ʒ�ִ���
	TThostFtdcVarietyType			VarietyType;  //Ʒ�����	
	TThostFtdcMarketType			MarketType;   //�г���־
	TThostFtdcProductIDType			ProductID;    //��Ʒ����	
};

///��ѯ�ʽ��˻�
struct CThostFtdcQryTradingAccountField
{
	TThostFtdcClientIDType			 ClientID;      //�ͻ���	
};

//��ѯ��ʷ�ʽ�
struct CThostFtdcQryHisCapitalField
{
	TThostFtdcClientIDType			 ClientID;      //�ͻ���	
	TThostFtdcDateType               StartDate;     //��ʼ����
	TThostFtdcDateType               EndDate;       //��������
};

///��ѯ���ʱ�֤����
struct CThostFtdcQryCostMarginFeeField
{
	TThostFtdcClientIDType			 ClientID;      //�ͻ���	
	TThostFtdcInstrumentIDType       InstID;		//��Լ����
};


//��ѯ��ʷ�ʽ�Ӧ��
struct CThostFtdcRspHisCapitalField
{
    TThostFtdcDateType              TradeDate;			//��������
	TThostFtdcMoneyType		        AvailCap;			//�����ʽ�
	TThostFtdcMoneyType		        Todayprofit;		//����ӯ��
	TThostFtdcMoneyType		        PosiMargin;			//�ֱֲ�֤��
	TThostFtdcMoneyType		        PickUpMargin;		//�����֤��
	TThostFtdcMoneyType		        StorageMargin;		//�ִ���֤��
	TThostFtdcMoneyType		        TradeFee;;			//����������
	TThostFtdcMoneyType		        TodayInOut;			//�����
	TThostFtdcMoneyType		        EtfTransferFee;		//ETF ������� 
	TThostFtdcMoneyType		        EtfTransferFeeFrzn; //ETF ������Ѷ���
	TThostFtdcMoneyType		        EtfCashBalance;     //T-1 ��ʵ���ֽ��� 
	TThostFtdcMoneyType		        EtfCashBalanceFrzn; //T ��Ԥ���ֽ���
};

///��ѯ���ʱ�֤����Ӧ��
struct CThostFtdcRspCostMarginFeeField
{
	TThostFtdcClientIDType			 ClientID;				//�ͻ���
	TThostFtdcInstrumentIDType       InstID;				//��Լ����
	TThostFtdcBuyOpenHandFee		 BuyOpenHandFee;        //����������
	TThostFtdcBuyOffsetHandFee       BuyOffsetHandFee;      //��ƽ��������
	TThostFtdcSellOpenHandFee		 SellOpenHandFee;       //������������
	TThostFtdcSellOffsetHandFee      SellOffsetHandFee;     //��ƽ��������
	TThostFtdcBuyMarginFee			 BuyMarginFee;          //��֤����
	TThostFtdcSellMarginFee          SellMarginFee;         //����֤����
	TThostFtdcSeatBuyMarginFee       SeatBuyMarginFee;      //ϯλ��֤����
	TThostFtdcSeatSellMarginFee      SeatSellMarginFee;     //ϯλ����֤����
};

///��ѯ�ʽ��˻�Ӧ��
struct CThostFtdcTradingAccountField
{
	TThostFtdcClientIDType	ClientID;	     //�ͻ���
	TThostFtdcMoneyType		AvailCap;        //�����ʽ�
	TThostFtdcMoneyType		Available;       //�����ʽ�
	TThostFtdcMoneyType		PosiMargin;      //�ֱֲ�֤��
	TThostFtdcMoneyType		BuyPosiMargin;   //��ֱֲ�֤��
	TThostFtdcMoneyType		SellPosiMargin;  //���ֱֲ�֤��
	TThostFtdcMoneyType		StorageMargin;   //�ִ���֤��
	TThostFtdcMoneyType		TotalFee;;       //��������
	TThostFtdcMoneyType		TotalFrozen;     //�ܶ����ʽ�
	TThostFtdcMoneyType		OrderFrozen;     //ί�ж���	
	TThostFtdcMoneyType		SpotSellFrozen;  //�ֻ���������
	TThostFtdcMoneyType		TodayIn;         //�������
	TThostFtdcMoneyType		TodayOut;        //���ճ���
	TThostFtdcMoneyType		LastFrozen;      //���ն����ʽ�
	TThostFtdcMoneyType		TotalFrozenFee;  //�ܶ���������	
	TThostFtdcMoneyType		PickUpMargin;    //�����֤��
	TThostFtdcMoneyType		MiddleMargin;    //�����ֱ�֤��
};

///��Ӧ��Ϣ
struct CThostFtdcRspInfoField
{
	TThostFtdcErrorIDType	ErrorID;	//�������
	TThostFtdcErrorMsgType	ErrorMsg;	//������Ϣ
};

///�������
struct CThostFtdcDepthMarketDataField
{
	TThostFtdcInstrumentIDType		InstID;       		//��Լ���� 
	TThostFtdcInstrumentNameType	Name;				//��Լ���� 
	TThostFtdcMarketNameType		MarketName;			//�г����ơ�00���ֻ��г�����01��tnԶ���г�����10�����ӣ���11������������ 
	TThostFtdcPriceType				PreSettle;			//����� 
	TThostFtdcPriceType				PreClose;			//������ 
	TThostFtdcPriceType				Open;				//���̼� 
	TThostFtdcPriceType				High;				//��߼� 
	TThostFtdcPriceType				Low;				//��ͼ� 
	TThostFtdcPriceType				Last;				//���¼� 
	TThostFtdcPriceType				Close;				//���̼� 
	TThostFtdcPriceType				Bid1;				//������һ 
	TThostFtdcVolumeType			BidLot1;			//�������һ,�������� 
	TThostFtdcPriceType				Ask1;				//�������һ 
	TThostFtdcVolumeType			AskLot1;			//�������һ 
	TThostFtdcPriceType				Bid2;				//�����۶� 
	TThostFtdcVolumeType			BidLot2;			//���������,�������� 
	TThostFtdcPriceType				Ask2;				//������۶� 
	TThostFtdcVolumeType			AskLot2;			//��������� 
	TThostFtdcPriceType				Bid3;				//�������� 
	TThostFtdcVolumeType			BidLot3;			//���������,����������
	TThostFtdcPriceType				Ask3;				//��������� 
	TThostFtdcVolumeType			AskLot3;			//��������� 
	TThostFtdcPriceType				Bid4;				//�������� 
	TThostFtdcVolumeType			BidLot4;			//���������,����������
	TThostFtdcPriceType				Ask4;				//��������� 
	TThostFtdcVolumeType			AskLot4;			//��������� 
	TThostFtdcPriceType				Bid5;				//�������� 
	TThostFtdcVolumeType			BidLot5;			//��������� 
	TThostFtdcPriceType				Ask5;				//��������� 
	TThostFtdcVolumeType			AskLot5;			//��������� 
	TThostFtdcPriceType				Bid6;				//�������� 
	TThostFtdcVolumeType			BidLot6;			//��������� 
	TThostFtdcPriceType				Ask6;				//��������� 
	TThostFtdcVolumeType			AskLot6;			//��������� 
	TThostFtdcPriceType				Bid7;				//�������� 
	TThostFtdcVolumeType			BidLot7;			//��������� 
	TThostFtdcPriceType				Ask7;				//��������� 
	TThostFtdcVolumeType			AskLot7;			//��������� 
	TThostFtdcPriceType				Bid8;				//�����۰� 
	TThostFtdcVolumeType			BidLot8;			//��������� 
	TThostFtdcPriceType				Ask8;				//������۰� 
	TThostFtdcVolumeType			AskLot8;			//���������
	TThostFtdcPriceType				Bid9;				//�����۾� 
	TThostFtdcVolumeType			BidLot9;			//��������� 
	TThostFtdcPriceType				Ask9;				//������۾� 
	TThostFtdcVolumeType			AskLot9;			//��������� 
	TThostFtdcPriceType				Bid10;				//������ʮ 
	TThostFtdcVolumeType			BidLot10;			//�������ʮ 
	TThostFtdcPriceType				Ask10;				//�������ʮ 
	TThostFtdcVolumeType			AskLot10;			//�������ʮ 
	TThostFtdcVolumeType			Volume;				//�ɽ�����˫�ߣ� 
	TThostFtdcVolumeType			OpenInt;			//�ֲ�����˫�ߣ� 
	TThostFtdcPriceType				UpDown;				//�ǵ� 
	TThostFtdcMoneyType				Turnover;			//�ɽ��� 
	TThostFtdcPriceType				Settle;				//����� 
	TThostFtdcPriceType				Average;			//���� 
	TThostFtdcDateType				QuoteDate;			//�������� 
	TThostFtdcTimeType				QuoteTime;			//����ʱ�� 
	TThostFtdcWeightType			weight;				//�ɽ���˫�ߣ�����
	TThostFtdcPriceType				highLimit;			//��ͣ��
	TThostFtdcPriceType				lowLimit;			//��ͣ��
	TThostFtdcRateType				UpDownRate;			//�ǵ�����
};

///���뱨��
struct CThostFtdcInputOrderField
{
	TThostFtdcSeatIDType			SeatID;          //ϯλ��
	TThostFtdcClientIDType			ClientID;        //�ͻ���
	TThostFtdcClientIDType			TradeCode;       //���ױ���		
	TThostFtdcInstrumentIDType      InstID;          //��Լ����	
	TThostFtdcBsFlagType			BuyOrSell;       //��������
	TThostFtdcOffsetFlagType		OffsetFlag;      //��ƽ��־	
	TThostFtdcVolumeType			Amount;          //ί������		
	TThostFtdcPriceType				Price;           //ί�м۸�		
	TThostFtdcMarketIDType			MarketID;        //�г�����	
	TThostFtdcOrderRefType			OrderRef;		 //��������
	TThostFtdcLoginBatchType		SessionID;		 //�ỰID
	TThostFtdcHedgeFlagType			HedgeFlag;		 //Ͷ����־
	TThostFtdcCmdType				CmdType;		 //ָ������
	TThostFtdcIPAddressType	        LoginIp;		 //�ն�IP��ַ
	TThostFtdcMacAddressType        MacAddress;	     //Mac��ַ
};

///������ί��
struct CThostFtdcConditionOrderField
{
	TThostFtdcExchangeIDType	    ExchangeID;		 //����������
	TThostFtdcSeatIDType			SeatID;          //ϯλ��
	TThostFtdcClientIDType			ClientID;        //�ͻ���
	TThostFtdcClientIDType			TradeCode;       //���ױ���
	TThostFtdcMarketIDType			MarketID;        //�г�����
	TThostFtdcInstrumentIDType      InstID;          //��Լ����
	TThostFtdcBsFlagType			BuyOrSell;       //��������
	TThostFtdcOffsetFlagType		OffsetFlag;      //��ƽ��־
	TThostFtdcVolumeType			Amount;          //ί������
	TThostFtdcByteType				OrderType;       //ί������
	TThostFtdcByteType				MiddleFlag;      //�����ֱ�־
	TThostFtdcByteType				PriceFlag;       //����ί�м۸�����
	TThostFtdcPriceType				Price;           //Ԥί�м۸�	
	TThostFtdcPriceType				TrigPrice;	     //Ԥί�д����۸�
	TThostFtdcValidDay				ValidDay;		 //��Ч������
	TThostFtdcVolumnCheck			VolumnCheck;	 //��������־
	TThostFtdcOrderRefType			OrderRef;		 //��������
	TThostFtdcLoginBatchType		SessionID;		 //�ỰID
	TThostFtdcCmdType				CmdType;		 //ָ������
	TThostFtdcIPAddressType	        LoginIp;		 //�ն�IP��ַ
	TThostFtdcMacAddressType        MacAddress;	     //Mac��ַ
   
};

///������ί��Ӧ��
struct CThostFtdcRspConditionOrderField
{
	TThostFtdcSeatIDType			SeatID;				//ϯλ��
	TThostFtdcClientIDType			ClientID;		    //�ͻ���
	TThostFtdcClientIDType			TradeCode;			//���ױ���
	TThostFtdcExchangeIDType	    ExchangeID;			//����������
	TThostFtdcOrderRefType			ConditionOrderNo;	//���������
	TThostFtdcDateType				EntrustDate;	    //ί������
	TThostFtdcDateType				TradeDate;		    //��Ȼ����
	TThostFtdcTimeType				EntrustTime;        //ί��ʱ��
	TThostFtdcOrderStatusType		Status;				//������ί��״̬
	TThostFtdcMarketIDType			MarketID;			//�г�����
	TThostFtdcInstrumentIDType      InstID;				//��Լ����
	TThostFtdcBsFlagType			BuyOrSell;			//��������
	TThostFtdcOffsetFlagType		OffsetFlag;			//��ƽ��־
	TThostFtdcVolumeType			Amount;				//ί������
	TThostFtdcByteType				OrderType;			//����������
	TThostFtdcByteType				MiddleFlag;			//�����ֱ�־
	TThostFtdcByteType				PriceFlag;			//����ί�м۸�����
	TThostFtdcPriceType				Price;				//Ԥί�м۸�	
	TThostFtdcPriceType				TrigPrice;			//Ԥί�д����۸�
	TThostFtdcValidDay				ValidDay;			//��Ч������
	TThostFtdcVolumnCheck			VolumnCheck;		//��������־
	TThostFtdcOrderRefType			OrderRef;			//��������
	TThostFtdcLoginBatchType		SessionID;			//�ỰID
	TThostFtdcCmdType				CmdType;			//ָ������
};

///���뱨��Ӧ��
struct CThostFtdcRspInputOrderField
{
	TThostFtdcOrderRefType			LocalOrderNo;	 //���ر�����
	TThostFtdcOrderStatusType		Status;			 //ί��״̬
	TThostFtdcSeatIDType			SeatID;          //ϯλ��
	TThostFtdcClientIDType			ClientID;        //�ͻ���
	TThostFtdcClientIDType			TradeCode;       //���ױ���		
	TThostFtdcInstrumentIDType      InstID;          //��Լ����	
	TThostFtdcBsFlagType			BuyOrSell;       //��������
	TThostFtdcOffsetFlagType		OffsetFlag;      //��ƽ��־	
	TThostFtdcVolumeType			Amount;          //ί������		
	TThostFtdcPriceType				Price;           //ί�м۸�		
	TThostFtdcMarketIDType			MarketID;        //�г�����	
	TThostFtdcOrderRefType			OrderRef;		 //��������
	TThostFtdcLoginBatchType		SessionID;		 //�ỰID
	TThostFtdcRequestIDType			RequestID;		 //������
	TThostFtdcHedgeFlagType			HedgeFlag;		 //Ͷ����־
	TThostFtdcCmdType				CmdType;		 //ָ������
};


///����������
struct CThostFtdcConditionActionOrderField
{
	TThostFtdcClientIDType			ClientID;		    //�ͻ���
	TThostFtdcOrderRefType			ConditionOrderNo;	//���������
	TThostFtdcIPAddressType	        LoginIp;			//�ն�IP��ַ
	TThostFtdcMacAddressType        MacAddress;			//Mac��ַ
};

///����������Ӧ��
struct CThostFtdcRspConditionActionOrderField
{
	TThostFtdcClientIDType			ClientID;		    //�ͻ���
	TThostFtdcOrderRefType			ConditionOrderNo;	//���������
};

///��������ѯ
struct CThostFtdcConditionOrderQryField
{
	TThostFtdcClientIDType			ClientID;			//�ͻ���
	TThostFtdcOrderRefType			ConditionOrderNo;	//���������
	TThostFtdcOrderStatusType		Status;				//������״̬
	TThostFtdcInstrumentIDType		InstID;				//��Լ����
	TThostFtdcDateType				StartDate;		    //��ʼ����
	TThostFtdcDateType				EndDate;		    //��������
	TThostFtdcOrderRefType			OrderRef;			//��������
	TThostFtdcLoginBatchType		SessionID;			//�ỰID
	TThostFtdcCmdType				CmdType;			//ָ������
};

///��������ѯӦ��
struct CThostFtdcRspConditionOrderQryField
{
	TThostFtdcClientIDType			ClientID;			//�ͻ���
	TThostFtdcDateType				StartDate;		    //��ʼ����
	TThostFtdcDateType				EndDate;		    //��������
	TThostFtdcOrderRefType			ConditionOrderNo;	//���������
	TThostFtdcOrderRefType			LocalOrderNo;		//���ر�����
	TThostFtdcByteType				OrderType;			//ί�е�����
	TThostFtdcCSRCInvestorNameType	EntrustTypeName;	//ί����������
	TThostFtdcInstrumentIDType		InstID;				//��Լ����
	TThostFtdcBsFlagType			BuyOrSell;	        //��������
	TThostFtdcOffsetFlagType		OffSetFlag;		    //��ƽ��־
	TThostFtdcOrderStatusType		Status;				//������״̬
	TThostFtdcVolumeType			Amount;				//ί������
	TThostFtdcPriceType				Price;				//ί�м۸�
	TThostFtdcPriceType				TriggerPrice;		//�����۸�
	TThostFtdcTimeType				EntrustTime;        //ί��ʱ��
	TThostFtdcTimeType				TriggerTime;        //����ʱ��
	TThostFtdcRemark				ReasonDesc;			//ԭ��
	TThostFtdcDateType				EntrustDate;	    //ί������
	TThostFtdcDateType				ExpireDate;			//������
	TThostFtdcDateType				TriggerDate;		//������������
	TThostFtdcOrderRefType			OrderRef;			//��������
	TThostFtdcLoginBatchType		SessionID;			//�ỰID
	TThostFtdcCmdType				CmdType;			//ָ������
};

///�������ɽ���ѯ
struct CThostFtdcConditionOrderMatchField
{
	TThostFtdcClientIDType			ClientID;			//�ͻ���
	TThostFtdcInstrumentIDType		InstID;				//��Լ����
	TThostFtdcOrderRefType			ConditionOrderNo;	//���������
	TThostFtdcOrderRefType			LocalOrderNo;		//ί�б��
};

///�������ɽ���ѯӦ��
struct CThostFtdcRspConditionOrderMatchField
{
	TThostFtdcClientIDType			ClientID;			//�ͻ���
	TThostFtdcMatchNoType			MatchNo;            //�ɽ������
	TThostFtdcOrderRefType			ConditionOrderNo;	//���������
	TThostFtdcOrderRefType			LocalOrderNo;		//ί�б��
	TThostFtdcByteType				EntrustType;		//ί������
	TThostFtdcInstrumentIDType		InstID;				//��Լ����
	TThostFtdcBsFlagType			BuyOrSell;	        //��������
	TThostFtdcOffsetFlagType		OffSetFlag;		    //��ƽ��־
	TThostFtdcVolumeType			Amount;				//ί������
	TThostFtdcPriceType				Price;				//ί�м۸�
	TThostFtdcVolumeType			MatchVolumn;		//�ɽ�����
	TThostFtdcPriceType				MatchPrice;			//�ɽ��۸�
	TThostFtdcTimeType				MatchTime;			//�ɽ�ʱ��
};

///���������ر�
struct CThostFtdcOrderRtnField
{
	TThostFtdcOrderNoType			orderNo;		//ϵͳ������	
	TThostFtdcOrderRefType			localOrderNo;;	//���ر�����	
	TThostFtdcOrderStatusType		status;			//ί��״̬	
	TThostFtdcVolumeType			cancelQty;		//��������
};


///���뱨������
struct CThostFtdcInputOrderActionField
{
	TThostFtdcClientIDType		ClientID;         //�ͻ���
	TThostFtdcOrderRefType		LocalOrderNo;     //���ر�����
	TThostFtdcMarketIDType		MarketID;		  //�г�����
	TThostFtdcOrderRefType		OrderRef; 		  //��������
	TThostFtdcLoginBatchType	SessionID;		  //�Ự���
	TThostFtdcRequestIDType		RequestID;		  //������
	TThostFtdcIPAddressType	   LoginIp;			  //�ն�IP��ַ
	TThostFtdcMacAddressType   MacAddress;		  //Mac��ַ
};


///���뱨������Ӧ��
struct CThostFtdcRspInputOrderActionField
{
	TThostFtdcClientIDType		ClientID;         //�ͻ���
	TThostFtdcOrderRefType		localOrderNo;     //���ر�����
	TThostFtdcOrderStatusType	Status;			  //ί��״̬
	TThostFtdcMarketIDType		marketID;		  //�г�����
	TThostFtdcOrderRefType		OrderRef; 		  //��������
	TThostFtdcLoginBatchType	SessionID;		  //�Ự���
	TThostFtdcRequestIDType		RequestID;		  //������
};

///��������
struct CThostFtdcOrderActionField
{
	TThostFtdcOrderRefType		localOrderNo;     //���ر�����	
	TThostFtdcByteType			orderFlag;        //ί�б�־	
	TThostFtdcMarketIDType		marketID;		  //�г�����	
	TThostFtdcTraderIDType		traderID;         //��������Ա
	TThostFtdcOrderRefType		OrderRef; 		  //��������
	TThostFtdcLoginBatchType	SessionID;		  //�Ự���
	TThostFtdcTradeWayType      tradeWay;         //ί������	
};

///��ѯ�ɽ�Ӧ��
struct CThostFtdcTradeField
{
	TThostFtdcClientIDType			ClientID;          //�ͻ���
	TThostFtdcOrderNoType			OrderNo;           //ϵͳ������
	TThostFtdcMatchNoType			MatchNo;           //�ɽ������
	TThostFtdcInstrumentIDType		InstID;            //��Լ����
	TThostFtdcBsFlagType			BuyOrSell;	       //��������
	TThostFtdcOffsetFlagType		OffSetFlag;		   //��ƽ��־
	TThostFtdcPriceType				Price;             //�ɽ��۸�
	TThostFtdcVolumeType			Volume;            //�ɽ�����
	TThostFtdcMoneyType				Amount;            //�ɽ����	
	TThostFtdcByteType				Order_flag;        //ί������
	TThostFtdcDateType				MatchDate;		   //�ɽ�����
	TThostFtdcTimeType				MatchTime;         //�ɽ�ʱ��	
	TThostFtdcOrderRefType			LocalOrderNo;	   //���ر�����
	TThostFtdcMarketIDType			MarketID;	       //�г�����	
	TThostFtdcMoneyType				Trade_fee;         //������
	TThostFtdcByteType				Forceoffset_flag;  //ǿƽ��־	
	TThostFtdcVolumeType			Forcebatchnum;     //ǿƽ���κ�
	TThostFtdcTradeWayType			TradeWay;          //������־
	TThostFtdcHedgeFlagType			HedgeFlag;		   //Ͷ����־
	TThostFtdcLoginBatchType		SessionID;		   //�Ự���
	TThostFtdcOrderRefType			OrderRef; 		   //��������
};

///��ѯ��Լ
struct CThostFtdcQryInstrumentField
{
	TThostFtdcClientIDType	  ClientID;            //�ͻ���
	TThostFtdcContractIDType  ContractID;          //��Լ����
	TThostFtdcProductIDType   ProductID;           //��Ʒ����
};

///��ѯ����Ʒ��
struct CThostFtdcQryVarietyCodeField
{
	TThostFtdcClientIDType	  ClientID;            //�ͻ���
	TThostFtdcContractIDType  VarietyID;		   //����Ʒ�ִ���
	TThostFtdcProductIDType   ProductID;           //��Ʒ����
};

///��ѯ�ɽ�
struct CThostFtdcQryTradeField
{
	TThostFtdcClientIDType		ClientID;           //�ͻ���
	TThostFtdcMarketIDType		MarketID;			//�г�����
	TThostFtdcInstrumentIDType  InstID;				//��Լ����
	TThostFtdcOrderRefType		LocalOrderNo;		//���ر������
};

///��ѯ����
struct CThostFtdcQryOrderField
{
	TThostFtdcClientIDType		ClientID;           //�ͻ���
	TThostFtdcMarketIDType		MarketID;			//�г�����
	TThostFtdcOrderRefType		LocalOrderNo;		//���ر������
	TThostFtdcInstrumentIDType  InstID;				//��Լ����
	TThostFtdcHedgeFlagType		HedgeFlag;		    //Ͷ����־
	TThostFtdcCmdType			CmdType;		    //ָ������
	TThostFtdcLoginBatchType	SessionID;			//�Ự���
	TThostFtdcOrderRefType		OrderRef; 			//��������
};

///��ѯ����Ӧ��
struct CThostFtdcOrderField
{
	TThostFtdcClientIDType			ClientID;				//�ͻ���
	TThostFtdcOrderNoType			OrderNo;				//ϵͳ������	
	TThostFtdcOrderRefType			LocalOrderNo;			//���ر�����
	TThostFtdcMarketIDType			MarketID;				//�г�����
	TThostFtdcInstrumentIDType      InstID;					//��Լ����
	TThostFtdcBsFlagType			BuyOrSell;				//��������
	TThostFtdcOffsetFlagType		OffsetFlag;				//��ƽ��־
	TThostFtdcVolumeType			Amount;					//ί������	
	TThostFtdcPriceType				Price;					//ί�м۸�
	TThostFtdcVolumeType			MatchQty;				//�ɽ�����
	TThostFtdcOrderStatusType		Status;					//ί��״̬	
	TThostFtdcTimeType				EntrustTime;			//ί��ʱ��
	TThostFtdcByteType				Forceoffset_flag;		//ǿƽ��־
	TThostFtdcVolumeType			CancelQty;				//��������
	TThostFtdcTimeType				CancelTime;				//����ʱ��
	TThostFtdcTradeWayType			TradeWay;				//��������
	TThostFtdcHedgeFlagType			HedgeFlag;				//Ͷ����־
	TThostFtdcLoginBatchType		SessionID;				//�Ự���
	TThostFtdcOrderRefType			OrderRef; 				//��������
	TThostFtdcCmdType				CmdType;				//ָ������
	TThostFtdcRequestIDType			RequestID;				//������
};

///��ѯͶ���ֲ߳�
struct CThostFtdcQryInvestorPositionField
{
	TThostFtdcClientIDType		ClientID;           //�ͻ���
	TThostFtdcMarketIDType		MarketID;			//�г�����
	TThostFtdcInstrumentIDType	InstID;				//��Լ����
};

///��ѯͶ���ֲ߳���ϸ
struct CThostFtdcQryInvestorPositionDetailField
{
	TThostFtdcClientIDType		ClientID;           //�ͻ���
	TThostFtdcDateType			QueryData;			//��ѯ����
};


///��ѯͶ���ֲ߳�Ӧ��
struct CThostFtdcInvestorPositionField
{
	TThostFtdcClientIDType			ClientID;					//�ͻ���
	TThostFtdcInstrumentIDType		InstID;						//��Լ����
	TThostFtdcMarketIDType			MarketID;					//�г�����
	TThostFtdcVolumeType			LongPosi;					//������ֲ���	
	TThostFtdcPriceType				LongPosiAvgPrice;			//��־���
	TThostFtdcVolumeType			ShortPosi;					//�������ֲ���	
	TThostFtdcPriceType				ShortPosiAvgPrice;			//���־���	
	TThostFtdcPriceType				LongOpenAvgPrice;			//�򿪾���
	TThostFtdcPriceType				ShortOpenAvgPrice;			//��������	
	TThostFtdcVolumeType			LongPosiFrozen;				//��ֲֶ���	
	TThostFtdcVolumeType			ShortPosiFrozen;			//���ֲֶ���	
	TThostFtdcVolumeType			LongPosiVol;				//��ֲ�����	
	TThostFtdcVolumeType			ShortPosiVol;				//���ֲ�����	
	TThostFtdcVolumeType			TodayLong;					//������	
	TThostFtdcVolumeType			TodayShort;					//��������	
	TThostFtdcVolumeType			TodayOffsetShort;			//������ƽ	
	TThostFtdcVolumeType			TodayOffsetLong;			//������ƽ	
	TThostFtdcVolumeType			LastLong;					//������ֲ�	
	TThostFtdcVolumeType			LastShort;					//�������ֲ�		
};

///��ѯͶ���ֲ߳���ϸӦ��
struct CThostFtdcInvestorPositionDetailField
{
	TThostFtdcClientIDType			ClientID;					//�ͻ���
	TThostFtdcCSRCInvestorNameType	ClientShortName;			//�ͻ����
	TThostFtdcDateType				Data;						//����
	TThostFtdcInstrumentIDType		InstID;						//��Լ����
	TThostFtdcBsFlagType			BuyOrSell;					//��������
	TThostFtdcVolumeType			Volumn;						//����
	TThostFtdcPriceType				Settle;						//�����
	TThostFtdcDateType				OpenFlagData;				//��������
	TThostFtdcMatchNoType			MatchNo;				    //�ɽ������	
};

///��ѯ���
struct CThostFtdcQryStorageField
{
	TThostFtdcVarietyIDType  VarietyID;            //����Ʒ�ִ���
	TThostFtdcClientIDType	 ClientID;			   //�ͻ���
};

//ETF�Ϲ�
struct CThostFtdcSubScriptionField
{
   TThostFtdcClientIDType	    ClientID;	     //�ͻ���
   TThostFtdcSeatIDType			SeatNo;		     //ϯλ��
   TThostFtdcEtfCodeType		EtfCode;         //�ƽ�ETF�������
   TThostFtdcInstrumentIDType	InstrumentID1;	 //��Լ����1
   TThostFtdcWeightType			weight1;		 //����1
   TThostFtdcInstrumentIDType	InstrumentID2;	 //��Լ����2
   TThostFtdcWeightType			weight2;		 //����2
   TThostFtdcInstrumentIDType	InstrumentID3;	 //��Լ����3
   TThostFtdcWeightType			weight3;		 //����3
   TThostFtdcInstrumentIDType	InstrumentID4;	 //��Լ����4
   TThostFtdcWeightType			weight4;		 //����4
   TThostFtdcInstrumentIDType	InstrumentID5;	 //��Լ����5
   TThostFtdcWeightType			weight5;		 //����5
   TThostFtdcWeightType			Totalweight;     //������
   TThostFtdcLoginBatchType		SessionID;		 //�ỰID
   TThostFtdcIPAddressType	    LoginIp;		 //�ն�IP��ַ
   TThostFtdcMacAddressType     MacAddress;	     //Mac��ַ
};

//ETF�깺
struct CThostFtdcApplyForPurchaseField
{
   TThostFtdcClientIDType	    ClientID;	     //�ͻ���
   TThostFtdcSeatIDType			SeatNo;		     //ϯλ��
   TThostFtdcEtfCodeType		EtfCode;         //�ƽ�ETF�������
   TThostFtdcInstrumentIDType	InstrumentID1;	 //��Լ����1
   TThostFtdcWeightType			weight1;		 //����1
   TThostFtdcInstrumentIDType	InstrumentID2;	 //��Լ����2
   TThostFtdcWeightType			weight2;		 //����2
   TThostFtdcInstrumentIDType	InstrumentID3;	 //��Լ����3
   TThostFtdcWeightType			weight3;		 //����3
   TThostFtdcInstrumentIDType	InstrumentID4;	 //��Լ����4
   TThostFtdcWeightType			weight4;		 //����4
   TThostFtdcInstrumentIDType	InstrumentID5;	 //��Լ����5
   TThostFtdcWeightType			weight5;		 //����5
   TThostFtdcWeightType			Totalweight;     //������
   TThostFtdcFundUnitType       fundunit;        //�ݶ�
   TThostFtdcLoginBatchType		SessionID;		 //�ỰID
   TThostFtdcIPAddressType	    LoginIp;		 //�ն�IP��ַ
   TThostFtdcMacAddressType     MacAddress;	     //Mac��ַ
};

//ETF���
struct CThostFtdcRedeemField
{
   TThostFtdcClientIDType	    ClientID;		 //�ͻ���
   TThostFtdcSeatIDType			SeatNo;		     //ϯλ��
   TThostFtdcEtfCodeType		EtfCode;         //�ƽ�ETF�������
   TThostFtdcFundUnitType       fundunit;        //�ݶ�
   TThostFtdcLoginBatchType		SessionID;		 //�ỰID
   TThostFtdcIPAddressType	    LoginIp;		 //�ն�IP��ַ
   TThostFtdcMacAddressType     MacAddress;	     //Mac��ַ
};

//ETF�˻���
struct CThostFtdcETFBingingField
{
	TThostFtdcClientIDType	       ClientID;		 //�ͻ���
	TThostFtdcEtfStockTxCodeType   StockTradeCode;	 //Ͷ����֤ȯ�˺�
	TThostFtdcEtfCodeType		   EtfCode;          //�ƽ�ETF�������
	TThostFtdcManagedUnitType      EtfManagedUnit;   //�йܵ�Ԫ
	TThostFtdcIPAddressType	       LoginIp;		     //�ն�IP��ַ
	TThostFtdcMacAddressType       MacAddress;	     //Mac��ַ
};

//ETF�˻����
struct CThostFtdcETFUnBingingField
{
	TThostFtdcClientIDType	        ClientID;		 //�ͻ���
	TThostFtdcEtfStockTxCodeType    StockTradeCode;	 //Ͷ����֤ȯ�˺�
	TThostFtdcEtfCodeType		    EtfCode;         //�ƽ�ETF�������
	TThostFtdcIPAddressType	       LoginIp;		     //�ն�IP��ַ
	TThostFtdcMacAddressType       MacAddress;	     //Mac��ַ
};

//ETF�˻��󶨽��״̬����
struct CThostFtdcETFBindingStatusField
{
	TThostFtdcClientIDType	        ClientID;		         //�ͻ���
	TThostFtdcEtfStockTxCodeType    StockTradeCode;	         //Ͷ����֤ȯ�˺�
	TThostFtdcEtfCodeType		    EtfCode;                 //�ƽ�ETF�������
	TThostFtdcBindingStatusType   	BindingStatus;	         //�󶨺ͽ��״̬
	TThostFtdcDateType				BindingDate;	         //������
	TThostFtdcOrderRefType		    BindingLocalOrderNo;	 //�󶨱��ر��
	TThostFtdcOrderRefType		    BindingEtfrevsn;	     //�󶨱�����ˮ
    TThostFtdcDateType				UnBindingDate;	         //�������
	TThostFtdcOrderRefType		    UnBindingLocalOrderNo;	 //��󶨱��ر��
	TThostFtdcOrderRefType		    UnBindingEtfrevsn;	     //��󶨱�����ˮ
};

//ETF�����꽻�ײ�ѯ 
struct CThostFtdcQryETFTradeDetailField
{
   TThostFtdcClientIDType	        ClientID;		 //�ͻ���
};

//ETF�������嵥��ѯ 
struct CThostFtdcQryETFPcfDetailField
{
   TThostFtdcDateType		StartDate;              //��ʼ����            
   TThostFtdcDateType       EndDate;				//��������
   TThostFtdcEtfCodeType    EtfCode;				//�ƽ�ETF�������
};

//ETF�������嵥��ѯӦ�� 
struct CThostFtdcETFPcfDetailField
{
	TThostFtdcDateType		    TradeDate;             //������            
	TThostFtdcEtfCodeType       EtfCode;			   //�ƽ�ETF�������
	TThostFtdcInstrumentIDType	InstrumentID1;	       //��Լ����1
	TThostFtdcMoneyType			TMoneydiff1;		   //T��Ԥ���ֽ���1
	TThostFtdcMoneyType			TPreMoneydiff1;		   //T-1��ʵ���ֽ���1
	TThostFtdcInstrumentIDType	InstrumentID2;	       //��Լ����2
	TThostFtdcMoneyType			TMoneydiff2;		   //T��Ԥ���ֽ���2
	TThostFtdcMoneyType			TPreMoneydiff2;		   //T-1��ʵ���ֽ���2
	TThostFtdcInstrumentIDType	InstrumentID3;	       //��Լ����3
	TThostFtdcMoneyType			TMoneydiff3;		   //T��Ԥ���ֽ���3
	TThostFtdcMoneyType			TPreMoneydiff3;		   //T-1��ʵ���ֽ���3
	TThostFtdcInstrumentIDType	InstrumentID4;	       //��Լ����4
	TThostFtdcMoneyType			TMoneydiff4;		   //T��Ԥ���ֽ���4
	TThostFtdcMoneyType			TPreMoneydiff4;		   //T-1��ʵ���ֽ���4
	TThostFtdcInstrumentIDType	InstrumentID5;	       //��Լ����5
	TThostFtdcMoneyType			TMoneydiff5;		   //T��Ԥ���ֽ���5
	TThostFtdcMoneyType			TPreMoneydiff5;		   //T-1��ʵ���ֽ���5
	TThostFtdcWeightType        minTradeWeight;	       //��С�깺�������
	TThostFtdcWeightType        TodayPurchaseMaxLimit; //�����깺����
	TThostFtdcWeightType        TodayRedeemMaxLimit;   //�����������
	TFtdcETFALLOWStatusType     TodayAllow;            //�����Ƿ������깺/��ر�־
	TThostFtdcWeightType        PreETFWeight;          //ÿ�ݻƽ�ETF�����Ӧ�Ļƽ�����
};


//ETF�����꽻�ײ�ѯӦ��
struct CThostFtdcETFTradeDetailField
{
	TThostFtdcClientIDType	      ClientID;			    //�ͻ���
    TThostFtdcEtfTradeType        tradeType;            //��������
	TThostFtdcOrderNoType		  OrderNo;				//����ETF���ױ��	
	TThostFtdcOrderRefType		  LocalOrderNo;			//����������
	TThostFtdcDateType			  RequestDate;			//��������
	TThostFtdcTimeType            RequestTime;          //����ʱ��
	TThostFtdcEtfCodeType	      EtfCode;              //�ƽ�ETF�������
	TThostFtdcInstrumentIDType	  InstrumentID1;	    //��Լ����1
	TThostFtdcWeightType		  weight1;				//����1
	TThostFtdcInstrumentIDType	  InstrumentID2;		//��Լ����2
	TThostFtdcWeightType		  weight2;				//����2
	TThostFtdcInstrumentIDType	  InstrumentID3;		//��Լ����3
	TThostFtdcWeightType		  weight3;				//����3
	TThostFtdcInstrumentIDType	  InstrumentID4;		//��Լ����4
	TThostFtdcWeightType		  weight4;				//����4
	TThostFtdcInstrumentIDType	  InstrumentID5;		//��Լ����5
	TThostFtdcWeightType		  weight5;				//����5
    TThostFtdcWeightType		  Totalweight;			//������
    TThostFtdcFundUnitType        fundunit;				//�ݶ�
	TThostFtdcDateType			  confirmDate;			//ȷ������
	TThostFtdcEtfTradeStatusType  tradestatus;          //����״̬
	TThostFtdcErrorMsgType	      ErrorMsg;	            //ʧ��ԭ�� 
	TThostFtdcRequestIDType		  RequestID;			//������
};

///���Ӧ��
struct CThostFtdcStorageField
{
	TThostFtdcClientIDType	 ClientID;			   //�ͻ���
	TThostFtdcVarietyIDType  VarietyID;            //����Ʒ�ִ���	
	TThostFtdcWeightType     totalStorage;		   //�������
	TThostFtdcWeightType     availableStorage;     //���ÿ��
	TThostFtdcWeightType     frozenStorage;	       //�ֻ�������	
	TThostFtdcWeightType     pendStorage;          //������
	TThostFtdcWeightType     todayBuy;             //��������
	TThostFtdcWeightType     todaySell;	           //��������
	TThostFtdcWeightType     todayDeposit;         //���մ���
	TThostFtdcWeightType     todayDraw;			   //�������
	TThostFtdcWeightType     todayBorrow;          //���ս���
	TThostFtdcWeightType     todayLend;	           //���ս��
	TThostFtdcWeightType     impawnStorage;        //��Ѻ���
	TThostFtdcWeightType     lawFrozen;            //���ɶ�����
	TThostFtdcWeightType     bankFrozen;           //���ж�����	
	TThostFtdcByteType       customType;           //�ͻ����
	TThostFtdcWeightType     storageCost;          //���ɱ�
	TThostFtdcWeightType     impawnFrozen;         //��Ѻ������
	TThostFtdcWeightType     transFrozen;          //����ҵ�񶳽���
};

///�г�״̬
struct CThostFtdcMarketStatusField
{
	TThostFtdcMktStatusType   	MktStatus;	//�г�״̬
	TThostFtdcMarketIDType    	marketID ;	//�г�����
	TThostFtdcExchCodeType   	ExchCode;	//����������
};

///��Լ״̬
struct CThostFtdcInstrumentStatusField
{
	TThostFtdcExchangeIDType		ExchangeID;			//����������
	TThostFtdcInstrumentIDType		InstrumentID;		//��Լ����
	TThostFtdcInstrumentStatusType	InstrumentStatus;	//��Լ����״̬
}; 

///ָ���ĺ�Լ
struct CThostFtdcSpecificInstrumentField
{
	TThostFtdcInstrumentIDType	InstrumentID; 		//��Լ����
};

///��ѯ����
struct CThostFtdcQryQuotationField
{
	TThostFtdcMarketIDType    	marketID ;			//�г�����
	TThostFtdcInstrumentIDType	InstrumentID;       //��Լ����
};

///�޸�����
struct CThostFtdcModifyPasswordField
{
	TThostFtdcOldPassword		OldPassword;		//ԭ����
	TThostFtdcNewPassword		NewPassword;		//������
};

///�޸�����Ӧ��
struct CThostFtdcModifyPasswordRsqField
{
	TThostFtdcClientIDType		 ClientID;			//�ͻ���
};

///���г����
struct CThostFtdcBOCMoneyIOField
{
	TThostFtdcTransFerType		TransFerType;		//��������ͣ�0����1���
	TThostFtdcTransFerAmount	TransFerAmount;		//ת�˽��
	TThostFtdcTradePassword		TradePassword;		//��������
	TThostFtdcClientIDType		ClientID;			//�ͻ���
};
///���г����Ӧ��
struct CThostFtdcBOCMoneyIORspField
{
	TThostFtdcClientIDType		 ClientID;			//�ͻ���
};

}	/// end of namespace KSGoldTradeAPI




#endif  ///__KSGOLDUSERAPISTRUCTEX__H__
